// components/ModelInfo.js
import React, { useEffect, useState } from 'react';

const ModelInfo = () => {
  const [modelsInfo, setModelsInfo] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:5000/models');
        const data = await response.json();
        setModelsInfo(data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div>
      {modelsInfo ? (
        <>
          <h1>Model Information</h1>
          {/* Display modelsInfo data in your Next.js component */}
          <pre>{JSON.stringify(modelsInfo, null, 2)}</pre>
          {/* Render your components or UI based on the fetched data */}
        </>
      ) : (
        <p>Loading...</p>
      )}
    </div>
  );
};

export default ModelInfo;
