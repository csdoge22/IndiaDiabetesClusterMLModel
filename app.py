# We will use Selenium for web scraping
import os
import shutil
import requests
from selenium import webdriver
from dotenv import load_dotenv

load_dotenv()

driver = webdriver.Firefox()
driver.get("https://www.google.com/search?safe=active&sca_esv=5149ab7345a4c72f&q=banana+fruit&udm=2&fbs=ABzOT_CWdhQLP1FcmU5B0fn3xuWpA-dk4wpBWOGsoR7DG5zJBsxayPSIAqObp_AgjkUGqekYoUzDaOcDDjQfK4KpR2OIjj43mhrQBsMJgHY2LSx-SUj4wz68xSZ8iYTfqgrdxb3MJvHOMODdIcpti-xYMckL_DuO7Mno3LlWlsnznPPjfINcnPSb3s0mY1_Udv3xmGYGwDe_3zR2JNQT7OndwaUM5c3nJw&sa=X&ved=2ahUKEwiylIfr7pSLAxVclokEHeCWGGQQtKgLegQICxAB&biw=2560&bih=1328&dpr=1")

images = driver.find_elements(by='xpath', value="//div[@style='height:180px']//img")

BASE_URL = os.getenv("BASE_URL")

for i in range(len(images)):
    src = images[i].get_attribute("src")
    
    response = requests.get(src, stream=True)
    with open(os.path.join(BASE_URL,f'banana{i}.png'), 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    
    del response