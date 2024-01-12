import fluence

# Initialize the Fluence client
client = fluence.Client()

# Create a new service
service = fluence.Service(client, "my-service")

# Define a function to handle incoming requests
@service.handler
def hello(request):
    return fluence.Response({"message": "Hello, world!"})

# Deploy the service
service.deploy()

# Wait for incoming requests
while True:
    request = service.receive()
    service.respond(request, hello(request))