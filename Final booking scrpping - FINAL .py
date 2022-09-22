#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from bs4 import BeautifulSoup as bs
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re


# In[2]:


# Original way of scrapping - the updated one is below
def scrape_reviews(hotel_linkname):
#Create empty lists to put in reviewers’ information as well as all of the positive & negative reviews 
    info = []
    #bookings.com reviews link
    #Original: https://www.booking.com/reviews/in/hotel/ramada-caravela-beach-resort.en-gb.html?page=1
    url = 'https://www.booking.com/reviews/sa/hotel/'+ hotel_linkname +'.html?page=' 
    page_number = 1
    #Use a while loop to scrape all the pages 
    while True:
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        page = session.get(url + str(page_number)) #retrieve data from server
        print(url + str(page_number))
        soup = bs(page.text, "html.parser") # initiate a beautifulsoup object using the html source and Python’s html.parser
        #print(soup)
        review_boxs = soup.find_all('li',{'class':'review_item clearfix'})
        if len(review_boxs) == 0:
            break
        #print(review_boxs)
        for review_box in review_boxs:
            #ratings
            rating = review_box.find('span',{'class':'review-score-badge'}).text.strip()
            #print(rating)
            #reviewer_info
            reviewer_name = review_box.find('p',{'class':'reviewer_name'}).text.strip()
            #print(reviewer_name)
            reviewer_country = review_box.find('span',{'itemprop':'nationality'}).text.strip()
            #print(reviewer_country)
            general_review = review_box.find('div',{'class':'review_item_header_content'}).text.strip()
            #print(general_review)
            # reviewer_review_times
            review_times = review_box.find('div',{'class':'review_item_user_review_count'}).text.strip()
            #print(review_times)
            # review_date
            review_date = review_box.find('p',{'class':'review_item_date'}).text.strip().strip('Reviewed: ')
            #print(review_date)
            # reviewer_tag
            reviewer_tag = review_box.find('ul',{'class':'review_item_info_tags'}).text.strip().replace('\n\n\n','')
            .replace('•',',').lstrip(', ')
            #print(reviewer_tag)
            # negative_review
            try:
                negative_review = review_box.find('p',{'class':'review_neg'}).text.strip('눉').strip() 
            except:
                negative_review = ""
            #print(negative_review)
            # positive_review
            try:
                positive_review = review_box.find('p',{'class':'review_pos'}).text.strip('눇').strip()
            except:
                positive_review = ""
            #print(positive_review)

            # append all info into one list
            info.append([hotel_linkname,rating,reviewer_name,reviewer_country,general_review, 
            review_times,review_date,reviewer_tag,negative_review,positive_review])

        # page change
        page_number +=1

    # create data frame
    info_df = pd.DataFrame(info,
    columns = ['Hotel_name','Reviewer_rating','reviewer_name','Country','Overall_review','Review_times','Review_date'
               ,'Review_tags','negative_review','positive_review'])
    info_df['Reviewer_rating'] = pd.to_numeric(info_df['Reviewer_rating'] )
    info_df['Review_times'] = pd.to_numeric(info_df['Review_times'].apply(lambda x:re.findall("\d+", x)[0]))
    info_df['Review_date'] = pd.to_datetime(info_df['Review_date'])
    
    return info_df


# In[3]:


def scrape_all_hotels_new():
    all_hotels_info_df = pd.DataFrame()
# paste the second page of the result here, delete the number after ofset = *** 
    headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko)
               Version/9.0.2 Safari/601.3.9'}
    url = 'https://www.booking.com/searchresults.html?label=gen173nr-1FCAEoggI46AdIM1gEaA-IAQGYATG4ARfIAQzYAQHoAQH4AQKIAgGoAgO4ApeWwJgGwAIB0gIkZjNlNjgxOGUtNjAxZi00ODQ0LWI5YzMtOGU2OGVkNGFmMzdh2AIF4AIB&sid=88ddd07c9547a5c8e3847a10177cd40e&aid=304142&sb=1&sb_lp=1&src=index&src_elem=sb&error_url=https%3A%2F%2Fwww.booking.com%2Findex.html%3Flabel%3Dgen173nr-1FCAEoggI46AdIM1gEaA-IAQGYATG4ARfIAQzYAQHoAQH4AQKIAgGoAgO4ApeWwJgGwAIB0gIkZjNlNjgxOGUtNjAxZi00ODQ0LWI5YzMtOGU2OGVkNGFmMzdh2AIF4AIB%26sid%3D88ddd07c9547a5c8e3847a10177cd40e%26sb_price_type%3Dtotal%26%26&ss=Saudi+Arabia&is_ski_area=&checkin_year=&checkin_month=&checkout_year=&checkout_month=&efdco=1&group_adults=2&group_children=0&no_rooms=1&b_h4u_keep_filters=&from_sf=1&ss_raw=saudi&ac_position=0&ac_langcode=en&ac_click_type=b&dest_id=186&dest_type=country&place_id_lat=23.8859&place_id_lon=45.0792&search_pageview_id=12f10a8b859300e4&search_selected=true&search_pageview_id=12f10a8b859300e4&ac_suggestion_list_length=5&ac_suggestion_theme_list_length=0&offset='
#   url = 'https://www.booking.com/searchresults.html?label=gen173nr-1DCAEoggI46AdIM1gEaA-IAQGYATG4ARfIAQzYAQPoAQH4AQKIAgGoAgO4AoO7sJgGwAIB0gIkNWZlZjNjMzItZmQwYS00MzQ1LThlMzAtMjRkMmU1ZTk3MGU52AIE4AIB&sid=68d76c977965f2b5cf9b108f055a032a&aid=304142&city=-3096384&srpvid=bcd14f52ebf20017&offset='
    offset=0
    # add the total number of the hotels appeared on your research page I put 10 to test the code only
    hotel_num = 10 
    while True: 
        print(url + str(offset))
        session = requests.Session()
        retry = Retry(connect=3, backoff_factor=0.5)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        response=session.get(url + str(offset),headers=headers)
        soup=bs(response.text, "html.parser")
        #print(soup)
        property_cards = soup.find_all('div', {'class' : 'd20f4628d0'})
        #print(len(property_cards))
        #print(property_cards)
        if len(property_cards) == 0:
            break

        for property_card in property_cards:
            print("************************************"+str(hotel_num)+"****************************************")
            link_tag = property_card.find('a', href=True)["href"] #source for how to fetch only href content https://stackabuse.com/guide-to-parsing-html-with-beautifulsoup-in-python/
            #print(link_tag)
            hotel_name = link_tag.split("/")[5]
            hotel_name = hotel_name.split(".")[0]
            hotel_city_tag = property_card.find('span', {'class' : 'f4bd0794db b4273d69aa'}).text.strip()
#             if (hotel_city_tag.find(",") != None )
#                 hotel_city = hotel_city_tag.split(",")[0]
#             else
#                 hotel_city = hotel_city_tag
#             print(hotel_city_tag)
            #print(hotel_name)
            if (hotel_city_tag.find(",") != -1 ):
                hotel_city = hotel_city_tag.split(",")[1].strip()
            else:
                hotel_city = hotel_city_tag
            print(hotel_city)
            hotel_info_df = scrape_reviews(hotel_name)
            all_hotels_info_df = all_hotels_info_df.append(hotel_info_df, ignore_index = True)
            hotel_num -= 1
            #try:
             #   number_of_reviews = property_card.find('div', {'class' : 'd8eab2cf7f c90c0a70d3 db63693c62'}).text.strip().strip('reviews')
            #except:
             #   number_of_reviews = '0'
            #print(number_of_reviews)
            
            #hotel_names_and_review_numbers.append([hotel_name,number_of_reviews])   
            if (hotel_num == 0):
                return all_hotels_info_df    
        offset +=25  

    # create data frame
    #hotel_names_and_review_numbers_df = pd.DataFrame(hotel_names_and_review_numbers, columns = ['Name','Review number'])
    #hotel_names_and_review_numbers['Review number'] = pd.to_numeric(hotel_names_and_review_numbers['Review number'] )
    
    return all_hotels_info_df    


# In[4]:


all_hotels_info_df = scrape_all_hotels_new()


# In[5]:


def show_data(df):
    print("The length of the dataframe is: {}".format(len(df)))
    print("Total NAs: {}".format(df.isnull().sum().sum()))
    return df


# In[6]:


show_data(all_hotels_info_df)


# In[7]:


# Below part is for changing language of the review, I just need to chang the language tag en to ar for arabic


# In[8]:


# https://www.booking.com/reviews/sa/hotel/dar-al-iman-intercontinental.en-gb.html?label=gen173rf-1FCA0oxAFCHGRhci1hbC1pbWFuLWludGVyY29udGluZW50YWxIM1gDaFCIAQGYAQm4ARfIAQzYAQHoAQH4AQOIAgGiAg5sb2NhbGhvc3Q6ODg4OKgCA7gCyJf_lwbAAgHSAiQzNThhYTg5Zi0zOGFjLTRlNTQtODczNC00MmQ3MDE2YWJmNWbYAgXgAgE&sid=da303ffd2cd1c2597519ccb7af4a792f&r_lang=ar&customer_type=total&order=featuredreviews
# https://www.booking.com/reviews/sa/hotel/dar-al-iman-intercontinental.en-gb.html?label=gen173rf-1FCA0oxAFCHGRhci1hbC1pbWFuLWludGVyY29udGluZW50YWxIM1gDaFCIAQGYAQm4ARfIAQzYAQHoAQH4AQOIAgGiAg5sb2NhbGhvc3Q6ODg4OKgCA7gCyJf_lwbAAgHSAiQzNThhYTg5Zi0zOGFjLTRlNTQtODczNC00MmQ3MDE2YWJmNWbYAgXgAgE&sid=da303ffd2cd1c2597519ccb7af4a792f&r_lang=en&customer_type=total&order=featuredreviews
# https://www.booking.com/reviews/sa/hotel/dar-al-iman-intercontinental.en-gb.html?label=gen173rf-1FCA0oxAFCHGRhci1hbC1pbWFuLWludGVyY29udGluZW50YWxIM1gDaFCIAQGYAQm4ARfIAQzYAQHoAQH4AQOIAgGiAg5sb2NhbGhvc3Q6ODg4OKgCA7gCyJf_lwbAAgHSAiQzNThhYTg5Zi0zOGFjLTRlNTQtODczNC00MmQ3MDE2YWJmNWbYAgXgAgE&sid=da303ffd2cd1c2597519ccb7af4a792f&customer_type=total&hp_nav=0&old_page=0&order=featuredreviews&page=2&r_lang=en&rows=75&


# In[9]:


#Save all scrapped data
all_hotels_info_df.to_csv("testrr.csv")


# In[12]:


#If the city name requied, update the original data scraper by adding the following

headers = {'User-Agent':'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_2) AppleWebKit/601.3.9 (KHTML, like Gecko) Version/9.0.2 Safari/601.3.9'}
url = 'https://www.booking.com/searchresults.html?label=gen173nr-1DCAMoxAFCCXJpeWFkaC1zYUgxWARoD4gBAZgBMbgBF8gBDNgBA-gBAYgCAagCA7gCiNaLmAbAAgHSAiQ2NTZmZTMyNy0xZTRmLTRlNmUtOGJmOC05ZDE5OWUxMWNhOWXYAgTgAgE&ssne=Saudi+Arabia&dest_type=country&offset=75&group_children=0&sb=1&checkout_year=&dest_id=186&b_h4u_keep_filters=&no_rooms=1&aid=304142&ss=Saudi+Arabia&error_url=https%3A%2F%2Fwww.booking.com%2Fcountry%2Fsa.html%3Faid%3D304142%26label%3Dgen173nr-1DCAMoxAFCCXJpeWFkaC1zYUgxWARoD4gBAZgBMbgBF8gBDNgBA-gBAYgCAagCA7gCiNaLmAbAAgHSAiQ2NTZmZTMyNy0xZTRmLTRlNmUtOGJmOC05ZDE5OWUxMWNhOWXYAgTgAgE%26sid%3D68d76c977965f2b5cf9b108f055a032a%26&src=country&src_elem=sb&checkin_year=&ssne_untouched=Saudi+Arabia&checkout_month=&group_adults=2&sid=68d76c977965f2b5cf9b108f055a032a&is_ski_area=0&checkin_month=&from_sf=1&sb_lp=1'
offset=0
hotel_num = 1 

print(url + str(offset))
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)
response=session.get(url + str(offset),headers=headers)
soup=bs(response.text, "html.parser")
#print(soup)
property_cards = soup.find_all('div', {'class' : 'd20f4628d0'})


for property_card in property_cards:
    print("************************************"+str(hotel_num)+"****************************************")
    link_tag = property_card.find('a', href=True)["href"] #source for how to fetch only href content https://stackabuse.com/guide-to-parsing-html-with-beautifulsoup-in-python/
    #print(link_tag)
    hotel_name = link_tag.split("/")[5]
    hotel_name = hotel_name.split(".")[0]
    print(hotel_name)
    hotel_city_tag = property_card.find('span', {'class' : 'f4bd0794db b4273d69aa'}).text.strip()
#    print(hotel_city_tag.find(","))
    if (hotel_city_tag.find(",") != -1 ):
        hotel_city = hotel_city_tag.split(",")[1].strip()
    else:
        hotel_city = hotel_city_tag
    print(hotel_city)

