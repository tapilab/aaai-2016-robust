import scrapy
import re
import time
import sys
from hansard.items import HansardItem


class HansardSpider(scrapy.Spider):
    name = 'hansard'

    def __init__(self, parliament_number=40, dst_dir="./", *args, **kwargs):
        super(HansardSpider, self).__init__(*args, **kwargs)
        self.allowed_domains = ['parl.gc.ca']
        self.base_url = 'http://www.parl.gc.ca/'
        self.housechamber_url = self.base_url + 'housechamberbusiness/'
        self.start_urls = [self.housechamber_url + 'ChamberSittings.aspx']
        self.parliament_number = int(parliament_number)
        self.dst_dir = dst_dir

    def parse(self, response):
        return self.parse_home(response)

    def parse_home(self, response):
        print response.url
        regex_parliament = r'href="(.*Parl=%d.*)"' % self.parliament_number
        html_links = response.xpath("//a[contains(@href, 'ChamberSittings')]").re(regex_parliament)
        sessions_links = [l.replace('&amp;', '&') for l in html_links]
        for sl in sessions_links:
            request = scrapy.Request(url=self.housechamber_url + sl,
                                     callback=self.parse_session)
            yield request

    def parse_session(self, response):
        print response.url
        if response.url.find('Key') == -1:
            # No Key parameter in the url, so we are looking for all the possible keys
            regex_parl_sess = re.compile(r".*Parl=([0-9]+).*Ses=([0-9]+).*")
            parl_number, sess_number = map(int, regex_parl_sess.match(response.url).group(1,2))
            links = response.xpath("//div[@id='ctl00_PageContent_divTabbedYears']//a/@href").extract()
            for l in links:
                request = scrapy.Request(url=self.housechamber_url + l,
                                         callback=self.parse_session)
                request.meta['hansard_parliament'] = parl_number
                request.meta['hansard_session'] = sess_number
                yield request
        else:
            # now, let's get every possible link and create a request to parse the debates on this date
            parl_number = response.meta['hansard_parliament']
            sess_number = response.meta['hansard_session']
            links_and_tags = response.xpath("//a[@class='PublicationCalendarLink']")
            hrefs = [x.xpath("@href")[0].extract() for x in links_and_tags]
            titles = links_and_tags.xpath("//a[@class='PublicationCalendarLink']/@title").extract()
            timestamps = [time.strptime(t.split(' (')[0], "%A %B %d, %Y") for t in titles]
            for href, tstamp in zip(hrefs, timestamps):
                # create id
                id = "%d-%d-%s" % (parl_number, sess_number, time.strftime("%Y%m%d", tstamp))
                request = scrapy.Request(url=self.base_url + href + '&xml=true',
                                         callback=self.parse_xml)
                request.meta['id'] = id
                yield request

    def parse_xml(self, response):
        print response.url
        xml_item = HansardItem()
        xml_item['id'] = response.meta['id']
        xml_item['content'] = response.body
        yield xml_item
