import time
import pathlib
import urllib.parse

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"

def get_page_source(url, timeout=30):
    """
    Fetch the HTML source of a webpage using undetected-chromedriver

    Args:
        url (str): The URL to scrape
        timeout (int): Maximum time to wait for page load in seconds

    Returns:
        str: HTML source of the page
    """
    safe_url = urllib.parse.quote(url, safe='')
    cache_path = pathlib.Path(f"webpage-cache/{safe_url}.html")
    if cache_path.exists():
        return cache_path.read_text()

    try:
        # Initialize the driver
        options = uc.ChromeOptions()
        options.headless = True
        options.page_load_strategy = 'eager'
        options.add_argument(f"--user-agent={user_agent}")

        driver = uc.Chrome(options=options, use_subprocess=False, no_sandbox=True)

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
        time.sleep(2)

        # Get the page source
        page_source = driver.page_source

        # Cache the page source
        cache_path.write_text(page_source)

        return page_source

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

    finally:
        try:
            driver.quit()
        except Exception as e:
            pass