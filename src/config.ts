// Configuration for API endpoints
export const config = {
  // API URL - will be automatically set based on environment
  apiUrl: import.meta.env.VITE_API_URL || 'https://oil-spill-backend.onrender.com',
  
  // Base URL for the application
  baseUrl: import.meta.env.BASE_URL || '/',
}; 