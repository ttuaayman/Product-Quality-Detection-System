import os

# List of required directories
directories = [
    "dataset/train",
    "dataset/test",
    "models",
    "results",
    "static/css",
    "static/js",
    "templates",
    "uploads",
    "scripts"
]

# Create directories if they don't exist
for directory in directories:
    os.makedirs(directory, exist_ok=True)

# Create empty essential files
essential_files = {
    "app.py": '''from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
''',
    "templates/index.html": "<h1>Welcome to Product Quality Detection</h1>",
    "templates/result.html": "<h1>Result Page</h1>",
    "templates/dashboard.html": "<h1>Dashboard</h1>"
}

for file_path, content in essential_files.items():
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

print("✅ Project structure initialized successfully!")
