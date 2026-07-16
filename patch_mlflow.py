files = ['src/evaluate.py', 'src/train.py']

old = 'with mlflow.start_run(run_name="test_evaluation", nested=True):'
new = '''try:
    with mlflow.start_run(run_name="test_evaluation", nested=True):
        pass
except:
    pass
# Patched:'''

for filepath in files:
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    if old in content:
        content = content.replace(old, new)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Patched {filepath}')
    else:
        print(f'String not found in {filepath}')