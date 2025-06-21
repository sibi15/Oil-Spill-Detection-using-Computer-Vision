import { Brain, Layers, Satellite } from 'lucide-react';

export interface ImageType {
  id: string;
  name: string;
  description: string;
  icon: typeof Brain | typeof Layers;
  uploadText: string;
}

export const imageTypes: ImageType[] = [
  {
    id: 'sar',
    name: 'SAR (Synthetic Aperture Radar)',
    description: 'Model trained with advanced augmentation techniques',
    icon: Satellite,
    uploadText: 'Upload Oil Spill Image'
  },
  // {
  //   id: 'sar-augmented',
  //   name: 'SAR with augmentation',
  //   description: 'SAR with advanced augmentation techniques',
  //   icon: Layers,
  //   uploadText: 'Upload SAR with augmentation Image'
  // },
  // {
  //   id: 'infrared',
  //   name: 'Infrared',
  //   description: 'Infrared analysis using custom model',
  //   icon: Brain,
  //   uploadText: 'Upload Infrared Image',
  // },
];