# Vercel Deployment Checklist

## Pre-Deployment Tasks

### ✅ Code Preparation
- [ ] All hardcoded Windows paths removed from `backend/app.py`
- [ ] Requirements.txt updated with cross-platform compatible versions
- [ ] Frontend API calls updated to use config-based URLs
- [ ] Large model files added to `.gitignore`
- [ ] Vercel configuration files created (`vercel.json`)

### ✅ Model Hosting Setup
- [ ] Upload `infrared_model.keras` to cloud storage (Google Drive, AWS S3, etc.)
- [ ] Upload `sar_model.keras` to cloud storage
- [ ] Get direct download URLs for both models
- [ ] Update `MODEL_URLS` in `backend/app.py` with actual URLs

### ✅ Environment Configuration
- [ ] Create Vercel account
- [ ] Set up environment variables in Vercel dashboard:
  - `VITE_API_URL`: Your Vercel app URL (e.g., `https://your-app.vercel.app/api`)

## Deployment Steps

### 1. Repository Setup
- [ ] Push code to GitHub repository
- [ ] Ensure `.gitignore` excludes large model files
- [ ] Verify all files are committed

### 2. Vercel Deployment
- [ ] Connect GitHub repository to Vercel
- [ ] Configure build settings:
  - Framework Preset: Vite
  - Build Command: `npm run build`
  - Output Directory: `dist`
  - Install Command: `npm install`
- [ ] Deploy the application

### 3. Post-Deployment Verification
- [ ] Frontend loads correctly at Vercel URL
- [ ] API endpoint `/api/predict` is accessible
- [ ] Models download successfully on first use
- [ ] Image upload and processing works
- [ ] Results display correctly

## Troubleshooting

### If Models Don't Download
- [ ] Check model URLs in `backend/app.py`
- [ ] Verify URLs are publicly accessible
- [ ] Check Vercel function logs for download errors

### If API Calls Fail
- [ ] Verify CORS configuration
- [ ] Check API URL in frontend config
- [ ] Test API endpoint directly

### If Build Fails
- [ ] Check Python version compatibility
- [ ] Verify all dependencies in requirements.txt
- [ ] Check for missing files or imports

## Performance Optimization

### For Production
- [ ] Consider using a CDN for model files
- [ ] Implement image compression before upload
- [ ] Add error handling for large files
- [ ] Monitor Vercel function execution time

### Monitoring
- [ ] Set up Vercel analytics
- [ ] Monitor API response times
- [ ] Track model download success rates
- [ ] Monitor memory usage

## Security Considerations

- [ ] Validate file uploads
- [ ] Implement rate limiting if needed
- [ ] Secure model download URLs
- [ ] Add proper error handling

## Backup Plan

If Vercel deployment fails due to model size or other limitations:

1. **Alternative Platforms**:
   - Railway (supports larger files)
   - Render (good for Python apps)
   - Heroku (with proper configuration)

2. **Architecture Changes**:
   - Split frontend and backend into separate deployments
   - Use external API services for model inference
   - Implement model quantization for smaller file sizes

## Contact Information

For deployment issues:
- Check Vercel documentation
- Review function logs in Vercel dashboard
- Test locally before deploying 