import json


def extract_samples(source, words, no_answer, max_topics, max_paragraphs, max_qas):
    def filter_topic(topic):
        return any(w in topic['title'] for w in words) if words else True

    def filter_qa(qa):
        if no_answer is None:
            return True
        else:
            is_impossible = qa['is_impossible'] if 'is_impossible' in qa else None
            no_answers = not bool(qa['answers'])
            return no_answer == (is_impossible or no_answers)

    def filter_qas(qas):
        return [qa for qa in qas if filter_qa(qa)][:max_qas]

    def transform_paragraph(paragraph):
        return {
            'context': paragraph['context'],
            'qas': filter_qas(paragraph['qas']),
        }

    def transform_paragraphs(paragraphs):
        return [transform_paragraph(p) for p in paragraphs]

    def transform_topic(topic):
        return {
            'title': topic['title'],
            'paragraphs': [
                              p
                              for p in transform_paragraphs(topic['paragraphs'])
                              if p['qas']
                          ][:max_paragraphs]
        }

    def transform_topics(topics):
        return [
            transform_topic(topic)
            for topic in topics
            if filter_topic(topic)
        ][:max_topics]

    return {
        'version': source['version'],
        'data': [topic for topic in transform_topics(source['data']) if topic['paragraphs']]
    }


def mkdir(filename):
    import os
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))


def export_extracted_samples(source_data_type, extracted_set_name, words=None, no_answer=None, target_data_type=None, max_topics=None, max_paragraphs=None, max_qas=None):
    in_filename = f'./data/squad/{source_data_type}-v2.0.json'
    with open(in_filename, "r") as input:
        source = json.load(input)
        extracted = extract_samples(source, words, no_answer, max_topics, max_paragraphs, max_qas)

    s_topics = source["data"]
    s_paragraphs = [p for t in s_topics for p in t['paragraphs']]
    s_qas = [qa for p in s_paragraphs for qa in p['qas']]

    x_topics = extracted["data"]
    x_paragraphs = [p for t in x_topics for p in t['paragraphs']]
    x_qas = [qa for p in x_paragraphs for qa in p['qas']]

    print(f'''
    Extracted {extracted_set_name} from {source_data_type}
        {len(x_topics)}/{len(s_topics)} topics, 
        {len(x_paragraphs)}/{len(s_paragraphs)} paragraphs, 
        {len(x_qas)}/{len(s_qas)} quas 
        from {source_data_type}
        with {", ".join(words)} in title
        no answer: {no_answer}
        ''')

    out_filename = f'./data/{extracted_set_name}/{target_data_type or source_data_type}-v2.0.json'
    mkdir(out_filename)
    with open(out_filename, "w") as output:
        json.dump(extracted, output, indent=2)


def list_topics(data_type):
    with open(f'./data/{data_type}-v2.0.json', "r") as fh:
        source = json.load(fh)
        for t in source["data"]:
            print(t['title'])


# export_extracted_samples('train', 'small',words=['Zelda', 'Hydrogen', 'Brain', 'Comput'], no_answer=True)
# export_extracted_samples('dev', 'small', words=['Comput', 'Steam', 'Force'], no_answer=True)
# export_extracted_samples('test', 'small',words=['Oxygen', 'Prime', 'Geology', 'Miscellaneous'], no_answer=True)

# export_extracted_samples('train', 'small-answered', words=['Zelda', 'Hydrogen', 'Brain', 'Comput'], no_answer=False)
# export_extracted_samples('dev', 'small-answered', words=['Comput', 'Steam', 'Force'], no_answer=False)
# export_extracted_samples('test', 'small-answered', words=['Oxygen', 'Prime', 'Geology', 'Miscellaneous'], no_answer=False)

# export_extracted_samples('train', 'tiny', words=['Comput'], no_answer=None)
# export_extracted_samples('dev', 'tiny', words=['Comput'], no_answer=None)
# export_extracted_samples('test', 'tiny', words=['Miscellaneous'], no_answer=None)

# export_extracted_samples('train', 'tiny-answered', words=['Comput'], no_answer=False)
# export_extracted_samples('dev', 'tiny-answered', words=['Comput'], no_answer=False)
# export_extracted_samples('test', 'tiny-answered', words=['Miscellaneous'], no_answer=False)

export_extracted_samples('train', 'debug-81', words=['Computer'], no_answer=False, target_data_type='train', max_topics=1, max_paragraphs=8, max_qas=1)
export_extracted_samples('train', 'debug-81', words=['Computer'], no_answer=False, target_data_type='dev', max_topics=1, max_paragraphs=8, max_qas=1)
export_extracted_samples('train', 'debug-81', words=['Computer'], no_answer=False, target_data_type='test', max_topics=1, max_paragraphs=8, max_qas=1)

export_extracted_samples('train', 'debug-11', words=['Computer'], no_answer=False, target_data_type='train', max_topics=1, max_paragraphs=1, max_qas=1)
export_extracted_samples('train', 'debug-11', words=['Computer'], no_answer=False, target_data_type='dev', max_topics=1, max_paragraphs=1, max_qas=1)
export_extracted_samples('train', 'debug-11', words=['Computer'], no_answer=False, target_data_type='test', max_topics=1, max_paragraphs=1, max_qas=1)
