/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // The deployed site opens on the Observatory workspace.
  async redirects() {
    return [{ source: "/", destination: "/observatory", permanent: false }];
  }
};

export default nextConfig;
