import json


def extract_and_write_us_gaap_keys(input_json, output_file_name):
    try:
        us_gaap = input_json.get("facts", {}).get("us-gaap", {})
        us_gaap_keys = list(us_gaap.keys())
        output_data = {"us_gaap_keys": us_gaap_keys}

        # Yeni JSON dosyası
        with open(output_file_name, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"Written in {output_file_name}.")
    except AttributeError:
        print("Error: No 'us-gaap' key found in the JSON.")


# Kullanım örneği:
def extract_keys_with_units(input_json, output_file_name):
    keys_with_units = []

    def search_for_units(obj, current_key=''):
        if isinstance(obj, dict):
            if 'units' in obj:
                keys_with_units.append(current_key)
            for key, value in obj.items():
                search_for_units(value, key)
        elif isinstance(obj, list):
            for item in obj:
                search_for_units(item, current_key)

    try:
        # Tüm JSON'u recursive olarak tara
        search_for_units(input_json)

        # Sonuçları yeni bir sözlüğe yerleştir
        output_data = {"keys_with_units": keys_with_units}

        with open(output_file_name, 'w') as f:
            json.dump(output_data, f, indent=4)

        print(f"'units' içeren anahtarlar written in {output_file_name}.")
    except Exception as e:
        print(f"Hata oluştu: {str(e)}")


with open('response.json', 'r') as file:
    input_json = json.load(file)
    extract_keys_with_units(input_json, "keys_with_units.json")

