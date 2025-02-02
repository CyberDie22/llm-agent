import time
import pathlib
import urllib.parse

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

def get_page_source(url, timeout=30):
    """
    Fetch the HTML source of a webpage using undetected-chromedriver

    Args:
        url (str): The URL to scrape
        timeout (int): Maximum time to wait for page load in seconds

    Returns:
        str: HTML source of the page
    """
    print(f"Fetching page source for {url}")

    try:
        # Initialize the driver
        options = uc.ChromeOptions()
        options.headless = True
        options.page_load_strategy = 'eager'
        options.add_argument(f"--user-agent={user_agent}")

        driver = uc.Chrome(options=options, use_subprocess=True, no_sandbox=True)

        # Set page load timeout
        driver.set_page_load_timeout(timeout)

        # Navigate to the URL
        driver.get(url)

        # Wait for the page to load (wait for body tag)
        try:
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
        except TimeoutException:
            print(f"Timeout waiting for page load after {timeout} seconds")

        # Add a small delay to ensure dynamic content loads
        time.sleep(5)

        # Get the page source
        page_source = driver.execute_script("return document.documentElement.outerHTML;")

        return page_source

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    finally:
        try:
            driver.quit()
        except Exception as e:
            pass