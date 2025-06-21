import React from 'react';
import { Shield, Clock, AlertTriangle } from 'lucide-react';

const EnvironmentalImpactSection: React.FC = () => {
  return (
    <section id="impact" className="py-12 px-6 bg-blue-50">
      <div className="container mx-auto">
        <h2 className="text-3xl font-bold mb-8 text-center text-blue-900">Environmental Impact & Early Detection</h2>
        
        <div className="max-w-4xl mx-auto">
          <p className="text-lg text-gray-700 mb-8 leading-relaxed text-center">
            Finding oil spills early helps protect beaches, marine life, and local communities. 
            When spills are spotted quickly, cleanup teams can stop the oil from spreading too far.
          </p>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-10">
            <div className="bg-white rounded-lg shadow-md p-6 transform transition-transform duration-300 hover:scale-105">
              <div className="bg-blue-100 p-3 rounded-full w-14 h-14 flex items-center justify-center mb-4">
                <Shield className="h-8 w-8 text-blue-700" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-blue-800">Protection</h3>
              <p className="text-gray-600">
                Early detection systems help safeguard critical marine ecosystems and coastal communities 
                from devastating environmental damage.
              </p>
            </div>
            
            <div className="bg-white rounded-lg shadow-md p-6 transform transition-transform duration-300 hover:scale-105">
              <div className="bg-blue-100 p-3 rounded-full w-14 h-14 flex items-center justify-center mb-4">
                <Clock className="h-8 w-8 text-blue-700" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-blue-800">Rapid Response</h3>
              <p className="text-gray-600">
                The 48-hour window after a spill is critical. Our technology enables teams to deploy 
                containment resources during this crucial period.
              </p>
            </div>
            
            <div className="bg-white rounded-lg shadow-md p-6 transform transition-transform duration-300 hover:scale-105">
              <div className="bg-blue-100 p-3 rounded-full w-14 h-14 flex items-center justify-center mb-4">
                <AlertTriangle className="h-8 w-8 text-blue-700" />
              </div>
              <h3 className="text-xl font-semibold mb-3 text-blue-800">Prevention</h3>
              <p className="text-gray-600">
                With up to 50% of spills being preventable, our detection algorithms can identify 
                high-risk scenarios before catastrophic events occur.
              </p>
            </div>
          </div>
          
          <div className="mt-12 bg-white rounded-lg shadow-md overflow-hidden">
            <div className="p-6">
              <h3 className="text-xl font-semibold mb-4 text-blue-800">How Our Technology Works</h3>
              <p className="text-gray-700 mb-4">
                Our system uses advanced Synthetic Aperture Radar (SAR) imagery to detect oil slicks 
                on water surfaces. The SAR technology can penetrate through clouds and operate in 
                darkness, providing 24/7 monitoring capabilities.
              </p>
              <p className="text-gray-700">
                The machine learning algorithms have been trained on thousands of historical oil spill 
                images, allowing them to distinguish between natural phenomena (like algal blooms) and 
                actual oil spills with 85-95% accuracy.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default EnvironmentalImpactSection;