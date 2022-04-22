#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date  : 2022/4/22 9:57 上午
# @File  : generate_data.py
# @Author: johnson
# @Desc  :  生成数据集

import os
import json
import yaml
from py2neo import Graph

ralations_dict = {
    "BRAND_POPULARITY_IS": "品牌定位",
    "BRAND_COUNTRY_IS": "品牌国家",
    "PRODUCT_TEXTURE_IS": "产品质地",
    "PRODUCT_BRAND_IS": "产品品牌",
}

def get_data(save_json_path='data.json', use_cache=True):
    """
    生成数据集，并保存到本地
    :param use_cache: 判断是否使用本地的json文件还是重新获取
    :return:
    """
    save_json_path = os.path.join(data_dir, save_json_path)
    # 连接neo4j数据库
    if os.path.exists(save_json_path) and use_cache:
        with open(save_json_path, 'r', encoding='utf-8') as f:
            triple_data = json.load(f)
    else:
        print(f"连接neo4j数据库,获取数据")
        graph = Graph(host='192.168.50.189', user='neo4j', password='welcome', name='neo4j', port=7687)
        # 存储查询到的三元组数据
        triple_data = []
        print(f"开始查询三元组数据")
        for rel, rel_name in ralations_dict.items():
            # 查询三元组
            print(f"查询三元组数据，关系名称：{rel_name}")
            data = graph.run("MATCH (n)-[r:" + rel + "]->(m) RETURN n,r,m").data()
            for d in data:
                # 将三元组存储到列表中
                triple_data.append({
                    "subject": d["n"]["name"],
                    "predicate": rel_name,
                    "object": d["m"]["name"]
                })
        # 保存数据, 如果目录不存在，就创建
        if not os.path.exists(os.path.dirname(save_json_path)):
            os.makedirs(os.path.dirname(save_json_path))
        with open(save_json_path, 'w', encoding='utf-8') as f:
            json.dump(triple_data, f, ensure_ascii=False)
    print("数据集大小:", len(triple_data))
    return triple_data

def generate_question_answer():
    """
    生成问答数据集
    :return:
    """
    triple_data = get_data()
    # 存储问答数据
    question_answer_data = []
    for triple in triple_data:
        # 将三元组转换成问答数据
        question_answer_data.append({
            "head_entity": triple["subject"],
            "question": triple["subject"] + "的" + triple["predicate"] + "是什么？",
            "answer": triple["object"]
        })
    print(f"生成的问答对数据集大小：{len(question_answer_data)}")
    train_data_num = int(len(question_answer_data) * 0.8)
    valid_data_num = int(len(question_answer_data) * 0.1)
    train_data, valid_data, test_data = question_answer_data[:train_data_num], question_answer_data[train_data_num:train_data_num + valid_data_num], question_answer_data[train_data_num + valid_data_num:]
    # 保存到json格式的文件中
    with open(os.path.join(data_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False)
    with open(os.path.join(data_dir, 'valid.json'), 'w', encoding='utf-8') as f:
        json.dump(valid_data, f, ensure_ascii=False)
    with open(os.path.join(data_dir, 'test.json'), 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False)
    print("生成问答数据集完成")
    return question_answer_data

def generate_kge_data():
    """
    生成kge数据集
    """
    triple_data = get_data()
    # 构造dataset.yaml文件， 主要需要5个文件，分别是entity_ids.del，relation_ids.del，train.del，valid.del，test.del
    dataset ={'files.entity_ids.filename': 'entity_ids.del', # id到实体的映射
              'files.entity_ids.type': 'map',   # 表明是一个map，\tab 隔开，2个字段，分别是id和实体
              'files.relation_ids.filename': 'relation_ids.del',   # 和实体一样，也是id到关系的映射
              'files.relation_ids.type': 'map',
              'files.test.filename': 'test.del',  #测试集，每行都是一个三元组，用tab隔开，头实体id，关系id，尾实体id
              # 'files.test.size': 152, # 测试集大小
              'files.test.split_type': 'test',
              'files.test.type': 'triples',   # 表明是一个三元组格式的文件，用tab隔开，头实体id，关系id，尾实体id
              'files.test_without_unseen.filename': 'test_without_unseen.del', # 暂时不需要
              # 'files.test_without_unseen.size': 152,
              'files.test_without_unseen.split_type': 'test',
              'files.test_without_unseen.type': 'triples',
              'files.train.filename': 'train.del',
              # 'files.train.size': 4565, #   训练集大小
              'files.train.split_type': 'train',
              'files.train.type': 'triples',
              'files.train_sample.filename': 'train_sample.del',
              # 'files.train_sample.size': 109,
              'files.train_sample.split_type': 'train',
              'files.train_sample.type': 'triples',
              'files.valid.filename': 'valid.del',
              # 'files.valid.size': 109,
              'files.valid.split_type': 'valid',
              'files.valid.type': 'triples',
              'files.valid_without_unseen.filename': 'valid_without_unseen.del',
              # 'files.valid_without_unseen.size': 109,
              'files.valid_without_unseen.split_type': 'valid',
              'files.valid_without_unseen.type': 'triples',
              'name': 'mydata',  # 数据集的名称
              # 'num_entities': 280,  #我们不设定，让程序自动获取实体数量和关系数量
              # 'num_relations': 112
              }
    dataset_yaml = {}
    dataset_yaml["dataset"] = dataset
    print("数据集的配置内容是: ",yaml.dump(dataset_yaml))
    # 保存到yaml文件
    with open(os.path.join(data_dir,"dataset.yaml"), 'w', encoding='utf-8') as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    # 生成id到实体的映射文件 entity_ids.del
    entity_ids = {}
    for triple in triple_data:
        if triple["subject"] not in entity_ids:
            entity_ids[triple["subject"]] = len(entity_ids)
        if triple["object"] not in entity_ids:
            entity_ids[triple["object"]] = len(entity_ids)
    id2entity = {v: k for k, v in entity_ids.items()}
    with open(os.path.join(data_dir,"entity_ids.del"), 'w', encoding='utf-8') as f:
        for k, v in id2entity.items():
            f.write(f"{k}\t{v}\n")
    # 生成id到关系的映射文件 relation_ids.del
    relation_ids = {}
    for triple in triple_data:
        if triple["predicate"] not in relation_ids:
            relation_ids[triple["predicate"]] = len(relation_ids)
    id2relation = {v: k for k, v in relation_ids.items()}
    with open(os.path.join(data_dir,"relation_ids.del"), 'w', encoding='utf-8') as f:
        for k, v in id2relation.items():
            f.write(f"{k}\t{v}\n")
    train_data_num = int(len(triple_data) * 0.8)
    valid_data_num = int(len(triple_data) * 0.1)
    train_data, valid_data, test_data = triple_data[:train_data_num], triple_data[train_data_num:train_data_num + valid_data_num], triple_data[train_data_num + valid_data_num:]
    # 生成训练集文件，三元组格式，train.del
    with open(os.path.join(data_dir,"train.del"), 'w', encoding='utf-8') as f:
        for triple in train_data:
            f.write(f"{entity_ids[triple['subject']]}\t{relation_ids[triple['predicate']]}\t{entity_ids[triple['object']]}\n")
    # 生成验证集文件，三元组格式，valid.del
    with open(os.path.join(data_dir,"valid.del"), 'w', encoding='utf-8') as f:
        for triple in valid_data:
            f.write(f"{entity_ids[triple['subject']]}\t{relation_ids[triple['predicate']]}\t{entity_ids[triple['object']]}\n")
    # 生成测试集文件，三元组格式，test.del
    with open(os.path.join(data_dir,"test.del"), 'w', encoding='utf-8') as f:
        for triple in test_data:
            f.write(f"{entity_ids[triple['subject']]}\t{relation_ids[triple['predicate']]}\t{entity_ids[triple['object']]}\n")
    print(f"完best_score成数据集的配置，生成的数据集目录是：{data_dir}，生成的文件包括：{os.listdir(data_dir)}")
if __name__ == '__main__':
    data_dir = "data/mydata"
    generate_kge_data()
    generate_question_answer()
