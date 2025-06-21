import React from 'react';

const HistoricalDataSection: React.FC = () => {
  return (
    <section className="py-12 px-6 bg-gray-50">
      <div className="container mx-auto">
        <h2 className="text-3xl font-bold mb-8 text-center text-blue-900">Historical Oil Spill Impact</h2>
        <div className="bg-white rounded-lg shadow-lg overflow-hidden">
          <div className="p-4 flex justify-center">
            <img 
              src="https://raw.githubusercontent.com/crazygamer9597/tmphst/refs/heads/main/estimated_cumulative_tonnes_oil_spill.png" 
              alt="Historical Oil Spill Impact Graph" 
              className="max-w-full h-auto rounded"
            />
          </div>
          <div className="p-6 bg-blue-50">
            <h3 className="text-xl font-semibold mb-4 text-blue-800">Understanding the Data</h3>
            <p className="text-gray-700 mb-4">
              This graph illustrates the cumulative impact of oil spills over time, measured in tonnes. 
              The visualization shows how early detection technology has helped reduce the volume of spills 
              in recent years, despite increasing maritime traffic.
            </p>
            <p className="text-gray-700">
              Our detection systems aim to identify potential spills before they reach critical mass, 
              allowing for more effective containment and reduced environmental damage.
            </p>
          </div>
        </div>
      </div>
    </section>
  );
};

export default HistoricalDataSection;