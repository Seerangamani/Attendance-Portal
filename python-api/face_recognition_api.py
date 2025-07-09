import os
import warnings
import logging
import time
from functools import lru_cache
import hashlib

# Suppress all unwanted messages at the very beginning
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')
logging.getLogger().setLevel(logging.ERROR)

from flask import Flask, request, jsonify
from flask_cors import CORS
import base64   
import cv2
import numpy as np
import face_recognition

# Additional OpenCV suppression
cv2.setLogLevel(0)

app = Flask(__name__)
CORS(app)

# Global cache for storing encodings
encoding_cache = {}

def decode_base64_to_image(base64_str):
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]

        missing_padding = len(base64_str) % 4
        if missing_padding:
            base64_str += '=' * (4 - missing_padding)

        image_bytes = base64.b64decode(base64_str)
        np_arr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        return image
    except Exception as e:
        return None

def resize_image_for_face_recognition(image, max_width=800):
    """Resize image to reduce processing time while maintaining face recognition accuracy"""
    height, width = image.shape[:2]
    
    if width > max_width:
        scale = max_width / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    return image

def get_image_hash(base64_str):
    """Generate hash for image to use as cache key"""
    return hashlib.md5(base64_str.encode()).hexdigest()

def get_cached_or_compute_encoding(base64_str, is_stored_image=False):
    """Get encoding from cache or compute if not cached"""
    image_hash = get_image_hash(base64_str)
    
    # Only cache stored images (not captured images which are always new)
    if is_stored_image and image_hash in encoding_cache:
        return encoding_cache[image_hash]
    
    # Decode and process image
    image = decode_base64_to_image(base64_str)
    if image is None:
        return None
    
    # Resize image for faster processing
    image = resize_image_for_face_recognition(image)
    
    # Convert to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get face encodings with optimized model
    encodings = face_recognition.face_encodings(
        rgb_image,
        num_jitters=1,  # Reduced from default 100 for speed
        model='small'   # Use smaller/faster model
    )
    
    # Cache the result if it's a stored image
    if is_stored_image and encodings:
        encoding_cache[image_hash] = encodings[0]
        return encodings[0]
    
    return encodings[0] if encodings else None

@app.route('/compare', methods=['POST'])
def compare_images():
    start_time = time.time()
    
    try:
        data = request.get_json(force=True)
        
        captured_base64 = data.get('capturedImage')
        stored_base64 = data.get('storedImage')

        if not captured_base64 or not stored_base64:
            return jsonify({'match': False, 'error': 'Missing image data'}), 400

        # Get encodings (with caching for stored image)
        captured_encoding = get_cached_or_compute_encoding(captured_base64, is_stored_image=False)
        stored_encoding = get_cached_or_compute_encoding(stored_base64, is_stored_image=True)

        if captured_encoding is None or stored_encoding is None:
            return jsonify({'match': False, 'error': 'Face not detected in one or both images'}), 200

        # Calculate distance and match
        face_distances = face_recognition.face_distance([stored_encoding], captured_encoding)
        distance = face_distances[0]
        
        # Convert distance to confidence percentage
        confidence = (1 - distance) * 100
        
        # Set tolerance
        TOLERANCE = 0.45
        
        # Use custom tolerance
        match = face_recognition.compare_faces([stored_encoding], captured_encoding, tolerance=TOLERANCE)[0]
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'match': bool(match),
            'confidence': round(confidence, 2),
            'distance': round(distance, 4),
            'tolerance_used': TOLERANCE,
            'processing_time': round(processing_time, 3),
            'debug_info': f"Distance: {distance:.4f}, Confidence: {confidence:.2f}%, Time: {processing_time:.3f}s"
        }), 200

    except Exception as e:
        return jsonify({'match': False, 'error': str(e)}), 500

@app.route('/compare_multiple', methods=['POST'])
def compare_multiple_faces():
    start_time = time.time()
    
    try:
        data = request.get_json(force=True)
        
        captured_base64 = data.get('capturedImage')
        stored_images = data.get('storedImages')
        employee_ids = data.get('employeeIds')

        if not captured_base64 or not stored_images:
            return jsonify({'match': False, 'error': 'Missing image data'}), 400

        # Get captured image encoding
        captured_encoding = get_cached_or_compute_encoding(captured_base64, is_stored_image=False)
        
        if captured_encoding is None:
            return jsonify({'match': False, 'error': 'Face not detected in captured image'}), 200

        # Process stored images in parallel-like manner
        results = []
        stored_encodings_list = []
        
        for i, stored_base64 in enumerate(stored_images):
            stored_encoding = get_cached_or_compute_encoding(stored_base64, is_stored_image=True)
            
            if stored_encoding is not None:
                stored_encodings_list.append(stored_encoding)
                
                # Calculate distance and confidence
                distance = face_recognition.face_distance([stored_encoding], captured_encoding)[0]
                confidence = (1 - distance) * 100
                
                results.append({
                    'employee_id': employee_ids[i] if employee_ids and i < len(employee_ids) else i,
                    'distance': round(distance, 4),
                    'confidence': round(confidence, 2)
                })

        if not results:
            return jsonify({'match': False, 'error': 'No valid faces found in stored images'}), 200

        # Find best match
        best_match = min(results, key=lambda x: x['distance'])
        TOLERANCE = 0.45
        
        processing_time = time.time() - start_time
        
        return jsonify({
            'match': best_match['distance'] < TOLERANCE,
            'best_match': best_match,
            'all_results': results,
            'tolerance_used': TOLERANCE,
            'processing_time': round(processing_time, 3),
            'cache_size': len(encoding_cache)
        }), 200

    except Exception as e:
        return jsonify({'match': False, 'error': str(e)}), 500

# Cache management endpoints
@app.route('/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the encoding cache"""
    global encoding_cache
    cache_size = len(encoding_cache)
    encoding_cache.clear()
    return jsonify({'message': f'Cache cleared. Removed {cache_size} entries.'}), 200

@app.route('/cache/status', methods=['GET'])
def cache_status():
    """Get cache status"""
    return jsonify({
        'cache_size': len(encoding_cache),
        'cache_keys': list(encoding_cache.keys())[:10]  # Show first 10 keys
    }), 200

if __name__ == '__main__':
    app.run(debug=False, port=5000)