import React from 'react';
import { Droplet, Clock, Target, Shield, SatelliteDish, Droplets } from 'lucide-react';

interface StatProps {
  value: string;
  description: string;
  icon: React.ElementType;
  color: string;
}

const Stat: React.FC<StatProps> = ({ value, description, icon: Icon, color }) => {
  return (
    <div className="flex flex-col items-center text-center p-6 bg-white/10 backdrop-blur-sm rounded-lg shadow-md transition-transform duration-300 hover:transform hover:scale-105">
      <div className={`${color} rounded-full p-3 mb-4`}>
        <Icon className="h-8 w-8 text-white" />
      </div>
      <h3 className="text-2xl md:text-3xl font-bold text-white mb-2">{value}</h3>
      <p className="text-sm text-gray-100">{description}</p>
    </div>
  );
};

const StatsPanel: React.FC = () => {
  return (
    <section className="py-12 px-6 bg-gradient-to-b from-blue-800 to-blue-900 text-white">
      <div className="container mx-auto">
        <h2 className="text-3xl font-bold mb-8 text-center">Why Our Project Makes a Difference?</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <Stat 
            icon={Droplets}
            color="bg-blue-500"
            value="1M+" 
            description="gallons of oil spilled annually in marine environments" 
          />
          <Stat 
            icon={SatelliteDish}
            color="bg-green-500"
            value="85-95%" 
            description="accuracy in satellite-based early detection systems" 
          />
          <Stat 
            icon={Clock}
            color="bg-yellow-500"
            value="48h" 
            description="critical window for effective containment measures" 
          />
          <Stat 
            icon={Shield}
            color="bg-red-500"
            value="Up to 50%" 
            description="of spills preventable through early detection" 
          />
        </div>
      </div>
    </section>
  );
};

export default StatsPanel;