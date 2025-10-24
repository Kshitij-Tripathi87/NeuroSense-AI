from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
from werkzeug.utils import secure_filename
import torch
import numpy as np
from config import MODEL_PATH, CLASSES, ALLOWED_EXTENSIONS, MAX_FILE_SIZE
from preprocessing import preprocess_image
from model import get_model

# ====================================
# NeuroScan AI - Flask API Server
# ====================================

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load model and configuration
print("\n" + "="*60)
print("  NEUROSCAN AI - Starting Server")
print("="*60)

print("\nLoading model...")
try:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

# Load image dimensions used during training
config_file = os.path.join(os.path.dirname(MODEL_PATH), 'model_config.json')
if os.path.exists(config_file):
    with open(config_file, 'r') as f:
        model_config = json.load(f)
        TARGET_SIZE = tuple(model_config['target_size'])
        print(f"✓ Model config loaded: Image size {TARGET_SIZE}")
        if 'trained_on' in model_config:
            print(f"✓ Model trained on: {model_config['trained_on']}")
else:
    TARGET_SIZE = (224, 224)
    print(f"⚠ Warning: model_config.json not found, using default size {TARGET_SIZE}")

print("="*60 + "\n")


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_tumor_info(tumor_type):
    """Get detailed information about tumor type"""
    tumor_info = {
        'glioma': {
            'name': 'Glioma',
            'description': 'Tumor arising from glial cells in the brain or spine. Can be low-grade or high-grade.',
            'severity': 'Moderate to Severe',
            'characteristics': 'Most common type of brain tumor. Can be malignant or benign.',
            'recommendation': 'Immediate consultation with neuro-oncologist required. May require surgery, radiation therapy, or chemotherapy.',
            'next_steps': [
                'Consult a neuro-oncologist immediately',
                'Get a comprehensive MRI with contrast',
                'Discuss treatment options (surgery, radiation, chemotherapy)',
                'Consider genetic testing of the tumor'
            ]
        },
        'meningioma': {
            'name': 'Meningioma',
            'description': 'Tumor arising from the meninges, the protective layers surrounding the brain and spinal cord.',
            'severity': 'Mild to Moderate',
            'characteristics': 'Usually benign and slow-growing. Accounts for about 30% of brain tumors.',
            'recommendation': 'Consult with neurosurgeon. Treatment depends on size, location, and symptoms. May require monitoring or surgical removal.',
            'next_steps': [
                'Consult a neurosurgeon',
                'Regular monitoring with MRI scans',
                'Discuss surgical options if symptomatic',
                'Assess quality of life impact'
            ]
        },
        'notumor': {
            'name': 'No Tumor Detected',
            'description': 'The scan appears to show healthy brain tissue with no visible tumor or abnormalities.',
            'severity': 'None',
            'characteristics': 'Normal brain tissue structure observed.',
            'recommendation': 'No immediate medical intervention required. Continue regular health check-ups as advised by your physician.',
            'next_steps': [
                'Maintain regular health check-ups',
                'Follow up if symptoms develop',
                'Continue healthy lifestyle habits',
                'Consult doctor if concerns arise'
            ]
        },
        'pituitary': {
            'name': 'Pituitary Adenoma',
            'description': 'Tumor in the pituitary gland, which controls hormone production. Usually benign.',
            'severity': 'Mild to Moderate',
            'characteristics': 'Often benign and slow-growing. May affect hormone levels and cause various symptoms.',
            'recommendation': 'Consult with endocrinologist and neurosurgeon. May require medication or surgical intervention depending on size and hormone levels.',
            'next_steps': [
                'Consult an endocrinologist',
                'Get hormone level testing',
                'Discuss medication options',
                'Consider surgery if indicated',
                'Regular monitoring of hormone levels'
            ]
        }
    }
    
    return tumor_info.get(tumor_type, {
        'name': 'Unknown',
        'description': 'Classification unavailable',
        'severity': 'Unknown',
        'characteristics': 'Please consult with a medical professional',
        'recommendation': 'Seek immediate medical consultation',
        'next_steps': ['Consult a healthcare professional']
    })


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'target_image_size': TARGET_SIZE,
        'classes': CLASSES,
        'version': '1.0.0'
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    """Predict brain tumor type from uploaded image"""
    
    # Check if model is loaded
    if model is None:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    # Check if file is present
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    # Check if file is selected
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Check if file type is allowed
    if not allowed_file(file.filename):
        return jsonify({'error': f'Invalid file type. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Save file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess and predict
        processed_image = preprocess_image(filepath, TARGET_SIZE)
        processed_image = processed_image.unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(processed_image)
            probs = torch.softmax(outputs, dim=1)[0]
            predictions = probs.cpu().numpy()
        
        # Get predicted class
        predicted_class_idx = np.argmax(predictions)
        predicted_class = CLASSES[predicted_class_idx]
        confidence = float(predictions[predicted_class_idx]) * 100
        
        # Get all class probabilities
        all_predictions = {
            CLASSES[i]: float(predictions[i]) * 100 
            for i in range(len(CLASSES))
        }
        
        # Get tumor information
        tumor_info = get_tumor_info(predicted_class)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        # Prepare response
        response = {
            'success': True,
            'predicted_class': predicted_class,
            'confidence': round(confidence, 2),
            'all_predictions': {k: round(v, 2) for k, v in all_predictions.items()},
            'tumor_info': tumor_info,
            'image_size_used': TARGET_SIZE
        }
        
        return jsonify(response)
    
    except Exception as e:
        # Clean up file if it exists
        if 'filepath' in locals() and os.path.exists(filepath):
            os.remove(filepath)
        
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """Get model architecture and training information"""
    
    info = {
        'target_image_size': TARGET_SIZE,
        'classes': CLASSES,
        'num_classes': len(CLASSES),
        'model_type': 'CNN',
        'architecture': {
            'layers': [
                {'type': 'Conv2D', 'filters': 32, 'kernel_size': '3x3'},
                {'type': 'BatchNormalization'},
                {'type': 'MaxPooling2D', 'pool_size': '2x2'},
                {'type': 'Conv2D', 'filters': 64, 'kernel_size': '3x3'},
                {'type': 'BatchNormalization'},
                {'type': 'MaxPooling2D', 'pool_size': '2x2'},
                {'type': 'Conv2D', 'filters': 128, 'kernel_size': '3x3'},
                {'type': 'BatchNormalization'},
                {'type': 'MaxPooling2D', 'pool_size': '2x2'},
                {'type': 'Conv2D', 'filters': 256, 'kernel_size': '3x3'},
                {'type': 'BatchNormalization'},
                {'type': 'MaxPooling2D', 'pool_size': '2x2'},
                {'type': 'Flatten'},
                {'type': 'Dense', 'units': 512},
                {'type': 'Dropout', 'rate': 0.5},
                {'type': 'Dense', 'units': 256},
                {'type': 'Dropout', 'rate': 0.3},
                {'type': 'Dense', 'units': len(CLASSES), 'activation': 'softmax'}
            ]
        }
    }
    
    # Add training info if available
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            if 'trained_on' in config:
                info['trained_on'] = config['trained_on']
            if 'test_results' in config:
                info['test_results'] = config['test_results']
    
    return jsonify(info)


@app.route('/api/batch-predict', methods=['POST'])
def batch_predict():
    """Predict multiple images at once"""
    
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'files' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files')
    
    if not files or len(files) == 0:
        return jsonify({'error': 'No files selected'}), 400
    
    results = []
    
    for file in files:
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                processed_image = preprocess_image(filepath, TARGET_SIZE)
                processed_image = processed_image.unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    outputs = model(processed_image)
                    probs = torch.softmax(outputs, dim=1)[0]
                    predictions = probs.cpu().numpy()
                
                predicted_class_idx = np.argmax(predictions)
                predicted_class = CLASSES[predicted_class_idx]
                confidence = float(predictions[predicted_class_idx]) * 100
                
                results.append({
                    'filename': file.filename,
                    'predicted_class': predicted_class,
                    'confidence': round(confidence, 2),
                    'success': True
                })
                
                os.remove(filepath)
                
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'error': str(e),
                    'success': False
                })
    
    return jsonify({
        'total_images': len(files),
        'successful': len([r for r in results if r.get('success')]),
        'results': results
    })


# Error handlers
@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': f'File too large. Maximum size: {MAX_FILE_SIZE / (1024*1024)}MB'}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(e):
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    from config import HOST, PORT, DEBUG
    print(f"\n🚀 Server starting on http://{HOST}:{PORT}")
    print(f"📊 Model ready for predictions\n")
    app.run(host=HOST, port=PORT, debug=DEBUG)
