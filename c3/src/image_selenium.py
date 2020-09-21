'''
This is a web scraping function using selenium 
to obtain jpeg images from the USDA Forest Service's
Satellite Imagery database.

https://fsapps.nwcg.gov/afm/imagery.php
'''


import selenium
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from time import sleep  
import numpy as np
import pandas as pd
import urllib

def get_images(start_date, end_date, state='socal'):
    '''
    Function saves images to file location.
    Images are generated/saved on a daily basis
    
    PARAMETERS
    ----------
    start_date = string; format ex: '20200912' for September 12, 2020
    end_date = string;
    state = string; defaults to 'socal' 
        User can input 'socal' or 'norcal'

    RETURNS
    -------
    None
    '''
    
    dates = pd.date_range(start=start_date,end=end_date,freq='D').strftime('%Y-%m-%d')

    DRIVER_PATH = '/Users/chelseazaloumis/documents/scraping/chromedriver'

    wd = webdriver.Chrome(DRIVER_PATH)

    wd.get("https://fsapps.nwcg.gov/afm/imagery.php")
    print(f"Page Title is : {wd.title}")

    for date in dates:
        if state == 'socal':
            # Click on South California on the map
            wd.find_element_by_xpath('//body//area[4]').click()
        else: 
            wd.find_element_by_xpath('//body//area[5]').click()

        # Select "Terra MODIS Corrected Reflectance, True Color"
        satellite_dropdown = wd.find_element_by_xpath("//select[@name='layer']").click()
        #wd.select_by_visible_text('Terra MODIS Corrected Reflectance, True Color')
        wd.find_element_by_xpath('/html[1]/body[1]/table[1]/tbody[1]/tr[1]/td[1]/table[2]/tbody[1]/tr[1]/td[2]/div[1]/form[1]/table[1]/tbody[1]/tr[1]/td[2]/table[1]/tbody[1]/tr[1]/td[2]/select[1]/option[3]').click()

        # Find Image Date input box:
        image_date = wd.find_element_by_xpath("//input[@id='date']")

        # Deselect input (small x) to clear
        deselect = wd.find_element_by_xpath("//a[@class='the-datepicker__deselect-button']")
        deselect.click()

        # Enter input for date
        image_date.send_keys(f'{date}')

        # Exit calendar datepicker
        deselect_cal = wd.find_element_by_xpath("//a[@class='the-datepicker__button'][contains(text(),'×')]")
        deselect_cal.click()

        # Click "Generate Subset"
        enter = wd.find_element_by_xpath('//body//input[4]')
        enter.click()
        sleep(5)

        wd.switch_to_window(wd.window_handles[1])

        wd.get(wd.current_url)
        print(f"Page Title is : {wd.title}")
        #sleep(15)

        url = wd.find_element_by_xpath("//html//body//img").get_attribute("src")
        urllib.request.urlretrieve(url, f"{date}.jpg")
        print(f'Image {date} saved.')
        wd.close()

        wd.switch_to_window(wd.window_handles[0])
        wd.get(wd.current_url)
        print(f"Page Title is : {wd.title}")

    # End loop

    wd.quit()

if __name__ == "__main__":
    # ex of how function formats start and end date
    # january_2014 = pd.date_range(start='20140101',end='20140131',freq='D').strftime('%Y-%m-%d')

    # Socal
    start_date = '20180101'
    end_date = '20200913'
    get_images(start_date, end_date)

    # Norcal
    tart_date = '20180101'
    end_date = '20200913'
    get_images(start_date, end_date, state='norcal')

