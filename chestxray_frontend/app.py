# ============================================================
# app.py - Complete Chest X-Ray AI Flask Application
# ============================================================

import os
import io
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from werkzeug.utils import secure_filename
from transformers import ViTModel
from datetime import datetime

# PDF Report imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

DISEASE_CLASSES = [
    "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
    "Emphysema", "Fibrosis", "Hernia", "Infiltration", "Mass",
    "Nodule", "Pleural_Thickening", "Pneumonia", "Pneumothorax"
]

OPTIMAL_THRESHOLDS = {
    "Atelectasis": 0.35, "Cardiomegaly": 0.30, "Consolidation": 0.25,
    "Edema": 0.30, "Effusion": 0.35, "Emphysema": 0.30,
    "Fibrosis": 0.25, "Hernia": 0.20, "Infiltration": 0.25,
    "Mass": 0.30, "Nodule": 0.35, "Pleural_Thickening": 0.25,
    "Pneumonia": 0.20, "Pneumothorax": 0.30
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== MODEL ====================
class ViTChestXray(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.vit.config.hidden_size, num_classes)
        )
    def forward(self, x):
        return self.classifier(self.vit(x).pooler_output)

model = ViTChestXray().to(device)
weights_path = os.path.join(MODEL_DIR, 'model_weights.pt')

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Model weights not found at {weights_path}. Please copy model_weights.pt from your training export.")

model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print("Model loaded successfully")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==================== HELPERS ====================
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(image_path):
    """Run inference and return structured results with threshold."""
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    
    results = []
    for disease, prob in zip(DISEASE_CLASSES, probs):
        thresh = OPTIMAL_THRESHOLDS[disease]
        results.append({
            'disease': disease,
            'probability': float(prob),
            'threshold': thresh,  # FIXED: include threshold
            'detected': prob > thresh,
            'severity': 'high' if prob > 0.7 else 'moderate' if prob > thresh else 'low'
        })
    
    results.sort(key=lambda x: x['probability'], reverse=True)
    return results

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    # Save with unique filename
    filename = secure_filename(file.filename)
    base, ext = os.path.splitext(filename)
    counter = 1
    unique_filename = filename
    while os.path.exists(os.path.join(UPLOAD_FOLDER, unique_filename)):
        unique_filename = f"{base}_{counter}{ext}"
        counter += 1
    
    filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
    file.save(filepath)
    print(f"Saved file to: {filepath}")
    
    try:
        results = predict_image(filepath)
        detected = [r for r in results if r['detected']]
        
        return render_template('result.html',
                             results=results,
                             detected=detected,
                             num_detected=len(detected),
                             filename=unique_filename,
                             has_findings=len(detected) > 0)
    except Exception as e:
        import traceback
        print(f"Error during prediction: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    
    results = predict_image(filepath)
    
    return jsonify({
        'success': True,
        'predictions': results,
        'detected_diseases': [r['disease'] for r in results if r['detected']],
        'num_detected': sum(1 for r in results if r['detected'])
    })

@app.route('/report/<filename>')
def generate_report(filename):
    """Generate professional PDF medical report."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    
    # Re-run prediction
    results = predict_image(filepath)
    detected = [r for r in results if r['detected']]
    
    # Create PDF in memory
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1e40af'),
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("CHEST X-RAY AI ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 20))
    
    # Metadata table
    meta_data = [
        ['Report ID:', f'CXR-{datetime.now().strftime("%Y%m%d-%H%M%S")}'],
        ['Date:', datetime.now().strftime("%B %d, %Y at %H:%M")],
        ['Model:', 'Vision Transformer (ViT-B/16)'],
        ['Test AUC:', '82.73%'],
        ['Status:', 'COMPLETED']
    ]
    meta_table = Table(meta_data, colWidths=[2*inch, 4*inch])
    meta_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 30))
    
    # Findings Summary
    story.append(Paragraph("CLINICAL FINDINGS", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    if detected:
        finding_style = ParagraphStyle(
            'FindingStyle',
            parent=styles['Normal'],
            textColor=colors.HexColor('#dc2626'),
            fontSize=12,
            spaceAfter=10
        )
        story.append(Paragraph(f"<b>{len(detected)} Potential Finding(s) Detected</b>", finding_style))
        for d in detected:
            story.append(Paragraph(f"• <b>{d['disease'].replace('_', ' ')}</b> — Confidence: {d['probability']*100:.1f}%", styles['Normal']))
    else:
        normal_style = ParagraphStyle(
            'NormalStyle',
            parent=styles['Normal'],
            textColor=colors.HexColor('#059669'),
            fontSize=12
        )
        story.append(Paragraph("No significant findings detected", normal_style))
    
    story.append(Spacer(1, 30))
    
    # Detailed Results Table
    story.append(Paragraph("DETAILED PROBABILITY ANALYSIS", styles['Heading2']))
    story.append(Spacer(1, 10))
    
    table_data = [['Disease', 'Probability', 'Threshold', 'Status']]
    for r in results:
        status = 'DETECTED' if r['detected'] else 'Not Detected'
        table_data.append([
            r['disease'].replace('_', ' '),
            f"{r['probability']*100:.1f}%",
            f"{r['threshold']*100:.0f}%",
            status
        ])
    
    results_table = Table(table_data, colWidths=[2.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
    results_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e40af')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#f8fafc')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('PADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    story.append(results_table)
    
    # Disclaimer
    story.append(Spacer(1, 40))
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.HexColor('#6b7280'),
        alignment=1
    )
    story.append(Paragraph(
        "IMPORTANT: This AI analysis is for research and educational purposes only. "
        "It should not be used as a substitute for professional medical diagnosis. "
        "Always consult a qualified radiologist or physician for clinical decisions.",
        disclaimer_style
    ))
    
    doc.build(story)
    buffer.seek(0)
    
    return send_file(buffer, as_attachment=True,
                     download_name=f'CXR_Report_{datetime.now().strftime("%Y%m%d")}.pdf',
                     mimetype='application/pdf')

@app.route('/health')
def health():
    return jsonify({'status': 'healthy', 'model': 'chestxray-vit', 'device': str(device)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)