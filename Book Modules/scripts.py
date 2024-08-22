from pathlib import Path
import requests

def download_file_from_google_drive(file_id, destination):
    """Download a file from Google Drive given its file ID and save it to the destination path."""
    # check if the destination file exists
    my_file = Path(destination)
    if my_file.is_file():
        print(f'{destination} already exists - no file downloaded.')
    else:
        # URL to download the file
        URL = f"https://drive.google.com/uc?id={file_id}"
        
        # Make a request to get the file
        session = requests.Session()
        response = session.get(URL, params={'confirm': True}, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Write the file to the destination path
            with open(destination, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
            print(f"File downloaded successfully and saved to {destination}")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")

def download_Washington_DC_Image():
    '''Download the Washington DC Image adn Header'''
    file_id_header = '1EGiX2b00m5-GUyLyrouUM29Zxq2oQsY2'
    destination_header = 'WashingtonDC_Ref_156bands.hdr'
    download_file_from_google_drive(file_id_header, destination_header)

    file_id_image = '1UAetIUQC2hqHJJDP7s5pkuON3hhvuuz2'
    destination_image = 'WashingtonDC_Ref_156bands'
    download_file_from_google_drive(file_id_image, destination_image)