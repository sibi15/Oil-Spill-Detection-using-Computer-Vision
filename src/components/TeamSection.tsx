import React from 'react';
import { Users } from 'lucide-react';

interface TeamMemberProps {
  name: string;
  github: string;
}

const TeamMember: React.FC<TeamMemberProps> = ({ name, github }) => {
  return (
    <div className="bg-white rounded-lg shadow-md p-6 flex flex-col items-center text-center transform transition-all duration-300 hover:shadow-lg hover:translate-y-[-4px]">
      <div className="bg-blue-100 p-3 rounded-full w-14 h-14 flex items-center justify-center mb-4">
        <Users className="h-6 w-6 text-blue-700" />
      </div>
      <h3 className="text-lg font-semibold text-blue-900 mb-1">{name}</h3>
      <a
        href={`https://github.com/${github}`}
        target="_blank"
        rel="noopener noreferrer"
        className="text-gray-500 hover:text-blue-700"
      >
        @{github}
      </a>
    </div>
  );
};

const TeamSection: React.FC = () => {
  return (
    <section id="team" className="py-12 px-6 bg-gray-50">
      <div className="container mx-auto">
        <h2 className="text-3xl font-bold mb-8 text-center text-blue-900">Team Members</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-2xl mx-auto">
          <TeamMember name="Sowmya Sriram" github="sowmya0948" />
          <TeamMember name="Sibi Karthik C V" github="sibi15" />
        </div>
      </div>
    </section>
  );
};

export default TeamSection;