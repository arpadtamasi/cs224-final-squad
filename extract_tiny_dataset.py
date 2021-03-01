import json


def export_tiny(data_type, source, words):
    with open(f'./data/tiny/{data_type}-v2.0.json', "w") as tf:
        tiny_data = [
            p for p in source['data']
            if any(w in p['title'] for w in words)
        ]
        tiny_source = {
            'version': source['version'],
            'data': tiny_data
        }
        json.dump(tiny_source, tf, indent=2)


def copy_tiny(data_type, keywords):
    with open(f'./data/{data_type}-v2.0.json', "r") as fh:
        source = json.load(fh)
        topics = len(source["data"])
        print(f'loaded {topics} topics')
        export_tiny(data_type, source, keywords)


def list_topics(data_type):
    with open(f'./data/{data_type}-v2.0.json', "r") as fh:
        source = json.load(fh)
        for t in source["data"]:
            print(t['title'])

copy_tiny('dev', ['Comput', 'Steam', 'Force'])
copy_tiny('train', ['Zelda', 'Hydrogen', 'Brain', 'Comput'])
copy_tiny('test', ['Oxygen', 'Prime', 'Geology', 'Miscellaneous'])
