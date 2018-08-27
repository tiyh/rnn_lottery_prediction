
# -*- coding: utf-8 -*-

import urllib  
import urllib2
import re
import string
import MySQLdb
import time

class Ticai:   
    def __init__(self):
        self.number = []  
    def ticai_crawl(self,url):  
	req = urllib2.Request(url)    
        response = urllib2.urlopen(req) 
        self.deal_data(response.read().decode('utf8'))
    def deal_data(self,myPage): 
	myqiItems =  re.findall('<div.*?class="pig-uul-ll">(.*?)</div>\s</div>\s</div>',myPage,re.S)
        for qiItems in myqiItems:
            qiItems=qiItems+r"</div></div>"
            qishus = re.findall('<span>(.*?)</span>',qiItems,re.S)
            #print 'here'
            numlist = []
            period = ''
            time = ''
            for qishu in qishus:
                if len(qishu) == 8: 
                    period=qishu
                elif len(qishu) == 10: 
                    time=qishu
                elif len(qishu) == 2: 
                    numlist.append(qishu)
                else:
                    break 
            if len(numlist)==5 and qishu!='' and period!='': 
                numlist.sort()
                print ''.join(numlist)

                cursor = db.cursor()
                sql = "INSERT INTO rnnlottery(id, \
                    time, numbers) \
                    VALUES ('%d', '%s', '%s')" % \
                    (int(period), time, ''.join(numlist))
                try:
                    cursor.execute(sql)
                    db.commit()
                except:
                    db.rollback()
                    print "rollback"
		

if __name__ == '__main__':

    db = MySQLdb.connect("localhost", "root", "3664", "test", charset='utf8' )

    ticai = Ticai()

    url =r'http://wap.sdticai.com/index.php?g=Portal&m=Index&a=lottery_history&id=9&pageindex='
    for i in range(1,10): 
        ticai.ticai_crawl(url+str(i))
        time.sleep(0.03)
    db.close()
'''
create table rnnlottery (
  id varchar(8)  primary key,
  time varchar(10),
  numbers varchar(14) not null
);
'''

