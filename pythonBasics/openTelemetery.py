# Import required libraries
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

# Initialize tracer with your Honeycomb information
def initialize_tracer(service_name, api_key):
    # Create a resource that identifies your service
    resource = Resource.create({"service.name": service_name})
    
    # Set up the tracer provider with the resource
    provider = TracerProvider(resource=resource)
    
    # Configure the OTLP exporter with your Honeycomb API key
    otlp_exporter = OTLPSpanExporter(
        endpoint="https://api.honeycomb.io:443",
        headers={"x-honeycomb-team": api_key}
    )
    
    # Add the exporter to the provider
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
    
    # Set the global tracer provider
    trace.set_tracer_provider(provider)
    
    # Return a tracer instance
    return trace.get_tracer(__name__)

# Usage example
if __name__ == "__main__":
    # Replace with your actual service name and API key
    SERVICE_NAME = "my-python-app"
    API_KEY = "IZQLOgyIuRNeqkFeEfOX1c"
    
    # Initialize the tracer
    tracer = initialize_tracer(SERVICE_NAME, API_KEY)
    
    # Create and record spans
    with tracer.start_as_current_span("main_function") as parent:
        # Your application code here
        print("Hello, Telemetry!")
        
        # Create a child span
        with tracer.start_as_current_span("child_operation") as child:
            print("Performing some operation...")
            # Add custom attributes to your span
            child.set_attribute("operation.value", 42)
            child.set_attribute("operation.name", "important calculation")