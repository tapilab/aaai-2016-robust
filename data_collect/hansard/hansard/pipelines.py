# -*- coding: utf-8 -*-

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html
import os

class HansardPipeline(object):
    def process_item(self, item, spider):
        id = item['id']
        with open(os.path.join(spider.dst_dir, id + '.xml'), 'w+') as xml_out:
            xml_out.write(item['content'])
        return item
