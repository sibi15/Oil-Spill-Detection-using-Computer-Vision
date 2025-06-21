import React from 'react';
import { Droplet } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="bg-gradient-to-r from-blue-900 to-teal-800 text-white py-4 px-6 shadow-md">
      <div className="container mx-auto flex flex-col md:flex-row justify-between items-center">
        <div className="flex items-center mb-4 md:mb-0">
          <Droplet className="h-8 w-8 mr-2 text-blue-300" />
          <h1 className="text-2xl font-bold">Oil Spill Detection using Computer Vision</h1>
        </div>
        <nav>
          <ul className="flex space-x-6">
            <li><a href="#upload" className="hover:text-blue-300 transition-colors duration-200">Upload</a></li>
            <li><a href="#impact" className="hover:text-blue-300 transition-colors duration-200">Impact</a></li>
            <li><a href="#team" className="hover:text-blue-300 transition-colors duration-200">Team</a></li>
          </ul>
        </nav>
      </div>
    </header>
  );
};

export default Header;