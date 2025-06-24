# Oil Spill Detection Backend

This is the backend service for the Oil Spill Detection application. It provides API endpoints for image processing and oil spill detection using machine learning models.

## Deployment

### Heroku Deployment

1. Create a Heroku account if you don't have one
2. Install Heroku CLI
3. Login to Heroku:
   ```bash
   heroku login
   ```

4. Create a new Heroku app:
   ```bash
   heroku create your-app-name
   ```

5. Set environment variables:
   ```bash
   heroku config:set GDRIVE_SAR_MODEL_ID=your_sar_model_id
   heroku config:set GDRIVE_INFRARED_MODEL_ID=your_infrared_model_id
   ```

6. Deploy the application:
   ```bash
   git push heroku main
   ```

7. Check the application status:
   ```bash
   heroku ps
   ```

8. View logs:
   ```bash
   heroku logs --tail
   ```

## Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python app.py
   ```

The application will be available at http://localhost:5000
