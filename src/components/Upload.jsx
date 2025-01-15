import React, { useState } from 'react';

const Upload = () => {
  const [image, setImage] = useState(null);

  // Handle file selection
  const handleFileChange = (event) => {
    const file = event.target.files[0]; // Get the first file
    if (file) {
      setImage(URL.createObjectURL(file)); // Create a preview URL
    }
  };

  // Remove the uploaded image
  const handleRemoveImage = () => {
    setImage(null); // Reset the image state
  };

  return (
    <div style={styles.container}>
      <div style={styles.content}>
        <h1 style={styles.heading}>Upload a Picture</h1>
        {!image && (
          <input 
            type="file" 
            accept="image/*" 
            onChange={handleFileChange} 
            style={styles.input} 
          />
        )}
        {image && (
          <div style={styles.previewContainer}>
            <div style={styles.imageWrapper}>
              <img 
                src={image} 
                alt="Uploaded Preview" 
                style={styles.imagePreview} 
              />
              <button 
                onClick={handleRemoveImage} 
                style={styles.closeButton}
              >
                &times;
              </button>
            </div>
            
          </div>
        )}
      </div>
    </div>
  );
};

const styles = {
  container: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    height: '100vh',
    backgroundColor: '#f4f4f9',
  },
  content: {
    textAlign: 'center',
    padding: '20px',
    borderRadius: '10px',
    backgroundColor: '#ffffff',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
  },
  heading: {
    fontSize: '24px',
    marginBottom: '20px',
    color: '#333333',
  },
  input: {
    padding: '10px',
    fontSize: '16px',
  },
  previewContainer: {
    marginTop: '20px',
  },
  previewHeading: {
    fontSize: '18px',
    marginBottom: '10px',
    color: '#555555',
  },
  imageWrapper: {
    position: 'relative',
    display: 'inline-block',
  },
  
    imagePreview: {
      maxWidth: '300px', // Set maximum width (e.g., 300 pixels)
      maxHeight: '300px', // Set maximum height (e.g., 300 pixels)
      borderRadius: '10px',
      boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
      objectFit: 'contain', // Ensures the image fits within the box
    },
    
  
  closeButton: {
    position: 'absolute',
    top: '10px',
    right: '10px',
    backgroundColor: '#ff4d4d',
    border: 'none',
    color: 'white',
    borderRadius: '50%',
    width: '30px',
    height: '30px',
    fontSize: '18px',
    cursor: 'pointer',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)',
  },
};

export default Upload;
