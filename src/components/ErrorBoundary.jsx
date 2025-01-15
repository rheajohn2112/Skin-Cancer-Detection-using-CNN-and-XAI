import React from 'react';
import { useRouteError } from 'react-router-dom';

const ErrorBoundary = () => {
  const error = useRouteError();
  
  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>Oops! Something went wrong.</h1>
      <p>{error.statusText || error.message}</p>
      <a href="/" style={{ textDecoration: "none", color: "blue" }}>
        Go Back Home
      </a>
    </div>
  );
};

export default ErrorBoundary;
