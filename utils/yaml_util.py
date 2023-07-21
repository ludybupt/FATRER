import ruamel.yaml

def load_yaml(filename):
    with open(filename, 'r') as file:
        conf = ruamel.yaml.safe_load(file)
    return conf