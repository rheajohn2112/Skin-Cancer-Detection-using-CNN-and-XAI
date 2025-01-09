from flask import Flask, request, redirect, url_for
import os

UPLOAD_FOLDER='./uploads'

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file1' not in request.files:
            return 'File not present'
        file1 = request.files['file1']
        path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
        file1.save(path)
        return path

        return 'ok'
    return '''
    <h1>Upload new File</h1>
    <form method="post" enctype="multipart/form-data">
      <input type="file" name="file1">
      <input type="submit">
    </form>
    '''

@app.route('/')
def hello_world():
    return 'Hello World'

if __name__ == '__main__':

    app.run(debug=True)