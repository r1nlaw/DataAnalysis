import requests
from bs4 import BeautifulSoup
import csv


def parse_page(url):
    response = requests.get(url).text
    soup = BeautifulSoup(response, 'lxml')
    items = soup.find_all(class_="lb-item")
    tovar_array = []

    for item in items:
        name_elem = item.find(class_="lbic-name").find("a")
        name = name_elem.get_text(strip=True) if name_elem else None

        price_elem = item.find(class_="lbic-price")
        price = price_elem.get_text(strip=True) if price_elem else None

        date_elem = item.find(class_="lbic-date")
        date = date_elem.get_text(strip=True) if date_elem else None

        href_elem = item.find("a", class_="img-mp")
        href = href_elem['href'] if href_elem else None
        if name and price and date and href:
            try:
                price_value = int(price.replace(' ', '').replace('₽', '').replace('\u2009', '').strip())
                if 100 <= price_value <= 1000000:
                    tovar_array.append({
                        'name': name,
                        'price': price_value,
                        'date': date,
                        'href': href
                    })
            except ValueError:
                continue

    return tovar_array

def save_to_csv(data, filename="output.csv"):
    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['name', 'price', 'date', 'href'])
        writer.writeheader()
        writer.writerows(data)

base_url = "https://simferopol.unibo.ru/category/4/biznes_i_promishlennost/"
filtered_items = parse_page(base_url)

save_to_csv(filtered_items)

print(f"Парсинг завершен. Сохранено {len(filtered_items)} записей.")
for item in filtered_items:
    print(item)