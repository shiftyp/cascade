/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  async rewrites() {
    return [
      {
        source: '/api/diversity/:path*',
        destination: 'http://localhost:8000/api/diversity/:path*', // FastAPI backend
      },
    ]
  },
}

module.exports = nextConfig