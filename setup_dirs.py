import os

directories = [
    'models',
    'services',
    'utils',
    'static/css',
    'static/js/components',
    'templates/auth',
    'templates/components',
    'templates/education',
    'instance'
]

for directory in directories:
    path = os.path.join(os.path.dirname(__file__), directory)
    os.makedirs(path, exist_ok=True)
    print(f"Created directory: {path}")
