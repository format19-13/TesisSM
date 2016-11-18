# coding=utf-8
import urllib2
import re
from lxml import html
from HTMLParser import HTMLParser
import requests
import sys
import pycurl
import cookielib
import mechanize
import getpass
import time

from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support import ui


def page_loaded(driver):
	return driver.find_element_by_tag_name("body") != None

#driver = webdriver.PhantomJS()#executable_path=r'/home/vero/dev/phantomjs-2.1.1-linux-x86_64/bin/phantomjs')

#driver.get("https://www.facebook.com/profile.php?id=100012688690284&sk=about&section=contact-info&pnref=about")

#print driver.find_elements_by_xpath("//*[contains(text(), 'Viudo')]")

#soup = BeautifulSoup(driver.page_source, "html.parser")
#print soup.prettify()

# Opening the web browser
driver = webdriver.PhantomJS()
driver.get("https://www.facebook.com/profile.php?id=100012688690284&sk=about&section=contact-info&pnref=about")
wait = ui.WebDriverWait(driver, 10)
wait.until(page_loaded)

print(driver.page_source)

 
driver.close()
