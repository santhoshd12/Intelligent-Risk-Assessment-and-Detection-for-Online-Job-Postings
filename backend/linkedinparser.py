import time
import logging
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

class LinkedInJobFetcher:
    def __init__(self, headless=True):
        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)



    def fetch(self, url):
        logger.info(f"Fetching LinkedIn job posting: {url}")
        self.driver.get(url)

        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "show-more-less-html__markup"))
            )
        except:
            logger.error("❌ Could not find job description")
            return None

        # click "See more" if present
        try:
            see_more = self.driver.find_element(By.CLASS_NAME, "artdeco-card__actions")
            if see_more.is_displayed():
                self.driver.execute_script("arguments[0].click();", see_more)
                time.sleep(2)
        except:
            pass

        soup = BeautifulSoup(self.driver.page_source, "html.parser")

        title = soup.find("h1")
        job_title = title.get_text(strip=True) if title else "N/A"

        desc_div = soup.find("div", class_="show-more-less-html__markup")
        job_description = desc_div.get_text("\n", strip=True) if desc_div else "N/A"

        return {
            "title": job_title,
            "description": job_description
        }

    def close(self):
        self.driver.quit()


if __name__ == "__main__":
    url = "https://www.linkedin.com/jobs/view/4288666059/"
    scraper = LinkedInJobFetcher()
    job_data = scraper.fetch(url)
    scraper.close()

    print("============================================================")
    print("📋 JOB POSTING FROM LINKEDIN")
    print("============================================================")
    print(f"🏷️ Title: {job_data['title']}")
    print("\n📖 Full Description:\n")
    print(job_data["description"])
