# Use an official Nginx runtime as a parent image
FROM nginx:alpine

# Copy the static files of your frontend app to the web server's root directory
COPY . /usr/share/nginx/html

# Expose the default HTTP port
EXPOSE 80

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
