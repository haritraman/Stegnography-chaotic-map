from flask import Flask, request, jsonify
import librosa
import soundfile as sf

app = Flask(__name__)

@app.route('/upload-audio', methods=['POST'])
def upload_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        y, sr = librosa.load(file, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        return jsonify({
            'filename': file.filename,
            'sample_rate': sr,
            'duration': duration
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
