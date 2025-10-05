/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    // Use internal communication for API calls - no SSL needed
    let apiUrl
    if (process.env.NODE_ENV === 'production') {
      // Internal communication between Fly.io processes
      apiUrl = 'http://cascade-collector.internal:8000'
    } else {
      apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
    }
    
    console.log('API URL being used:', apiUrl)
    
    return [
      {
        source: '/api/diversity/:path*',
        destination: `${apiUrl}/api/diversity/:path*`,
      },
    ]
  },
}

module.exports = nextConfig