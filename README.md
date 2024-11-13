# toxic-comment-analysis

## Setting up the repository

You should set up your own virtual environment on your laptop to run all the notebooks in this repository. You may do this in your preferred way, but a simple way to set this up would be:

1. Run `python -m venv venv` to create your virtual environment.
2. Run `source venv/bin/activate` to activate the virtual environment.
3. Run `pip install -r requirements.txt` to install the full list of requirements.

## Perspective API .env Setup Instructions
### Setting Up Your .env File

1. **Create a `.env` File**:
   - Create .env inside the LLM folder.

2. **Add Your API Key**:
   - Open the `.env` file and add your Perspective API key as follows:
     ```
     PERSPECTIVE_API_KEY=your_api_key_here
     ```
   - Replace `your_api_key_here` with the actual API key you received from the Perspective API.

3. **Save the File**:
   - Ensure the `.env` file is saved in UTF-8 format.

4. **Environment Variables**:
   - The application will automatically load the key when running toxicity detection functions.

### Getting an API Key
- If you don’t have an API key, you can obtain one by signing up for access on the [Perspective API website](https://perspectiveapi.com/).
