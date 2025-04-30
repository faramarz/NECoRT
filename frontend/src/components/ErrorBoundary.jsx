import React from 'react';

class ErrorBoundary extends React.Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error) {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error };
  }

  componentDidCatch(error, errorInfo) {
    // You can also log the error to an error reporting service
    console.error("Component Error:", error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      // You can render any custom fallback UI
      return (
        <div className="p-6 bg-red-50 text-red-700 rounded-lg">
          <h2 className="text-xl font-bold mb-2">Something went wrong</h2>
          <p className="mb-4">We're having trouble loading this component. Please try:</p>
          <ul className="list-disc ml-6 mb-4">
            <li>Refreshing the page</li>
            <li>Checking your API key in settings</li>
            <li>Making sure the backend server is running</li>
          </ul>
          <p className="text-sm border-t border-red-200 pt-2 mt-2">
            Error details: {this.state.error && this.state.error.toString()}
          </p>
          <button 
            className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 