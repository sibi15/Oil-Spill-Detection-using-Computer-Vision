import React from 'react';
import Header from './components/Header';
import StatsPanel from './components/StatsPanel';
import HistoricalDataSection from './components/HistoricalDataSection';
import ImageUploadSection from './components/ImageUploadSection';
import EnvironmentalImpactSection from './components/EnvironmentalImpactSection';
import TeamSection from './components/TeamSection';
import Footer from './components/Footer';
import './utils/animations.css';

function App() {
  return (
    <div className="min-h-screen bg-white flex flex-col">
      <Header />
      <main>
        <StatsPanel />
        <HistoricalDataSection />
        <ImageUploadSection />
        <EnvironmentalImpactSection />
        <TeamSection />
      </main>
      <Footer />
    </div>
  );
}

export default App;