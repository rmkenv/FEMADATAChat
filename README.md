# FEMA Data Analysis with Langchain and Google Generative AI

This project fetches and analyzes FEMA NFIP Claims data based on a given ZIP code, utilizing Langchain and Google Generative AI to process and interact with the data.

## Features

- Fetch FEMA NFIP Claims data using FEMA API.
- Analyze and display data in a tabular format.
- Perform calculations and generate insights using Langchain tools.
- Hide sensitive API keys using environment variables.

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/fema-data-analysis.git
    cd fema-data-analysis
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the project root and add your Google API key:

    ```plaintext
    GOOGLE_API_KEY=your_google_api_key_here
    ```

5. Add `.env` to your `.gitignore` to ensure it is not pushed to GitHub:

    ```plaintext
    .env
    ```

## Usage

1. Run the script:

    ```sh
    python script.py
    ```

2. Enter your ZIP code when prompted.

3. The script will fetch and display the FEMA data, and export it to a CSV file named `fema_data_<zip_code>.csv`.

## Google Colab

To run this project on Google Colab, follow these steps:

1. Open Google Colab in your browser: [Google Colab](https://colab.research.google.com/).

2. Create a new notebook.

3. Upload the `script.py` file to the Colab notebook by clicking on the folder icon on the left sidebar and then the upload button.

4. Install the required packages by running the following cell:

    ```python
    !pip install requests pandas ipython langchain langchain-google-genai python-dotenv
    ```

5. Upload your `.env` file to the Colab notebook.

6. Add the following code cell to load the environment variables:

    ```python
    from dotenv import load_dotenv
    import os

    load_dotenv()
    os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
    ```

7. Add a code cell to run the script:

    ```python
    !python script.py
    ```

8. Run the cells in the notebook to execute the script.

## Functions

The project includes several functions to analyze the FEMA data:

- **Total Building Damage Amount**: Calculates the total building damage amount.
- **Average Contents Damage Amount**: Calculates the average contents damage amount.
- **Most Recent Date of Loss**: Finds the most recent loss date.
- **Count Policies by Flood Zone**: Counts policies by flood zone.
- **Total Number of Claims**: Calculates the total number of claims.
- **Total Building and Contents Damage Amount**: Calculates the total sum of building and contents damage amounts.

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any changes.

## Contact

For any inquiries, please contact [Ryan](mailto:kmetzrm@gmail.com).

