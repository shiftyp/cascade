export interface GeographicData {
  totalHours: number
  latitudeBands: {
    arctic: { hours: number; percentage: number; target: number }
    temperate: { hours: number; percentage: number; target: number }
    tropical: { hours: number; percentage: number; target: number }
    antarctic: { hours: number; percentage: number; target: number }
  }
  hemispheres: {
    north: { hours: number; percentage: number; target: number }
    south: { hours: number; percentage: number; target: number }
    equatorial: { hours: number; percentage: number; target: number }
  }
  gridSquareData: Array<{
    grid: string
    latitude: number
    longitude: number
    hours: number
    intensity: number
  }>
  oceanPathPercentage: number
}

export interface DiversityMetrics {
  simpsonDiversityIndex: number
  hemisphereBalanceScore: number
  continentalCoverage: {
    northAmerica: boolean
    southAmerica: boolean
    europe: boolean
    africa: boolean
    asia: boolean
    oceania: boolean
    antarctica: boolean
  }
  overallDiversityScore: number
}

export interface RebalancingRecommendation {
  region: string
  currentPercentage: number
  targetPercentage: number
  deficit: number
  priorityMultiplier: number
  action: string
}

export interface QASample {
  id: string
  sessionId: string
  timestamp: string
  frequencyKhz: number
  band: string
  mode?: string
  sampleRate: number
  duration: number
  gridSquare?: string
  callsignHash?: string
  correlationId?: string
  snr?: number
  propagationMode?: string
  spaceWeatherCondition?: string
  fileSizeBytes: number
  hasWaterfall: boolean
  s3Path?: string
  metadata?: Record<string, any>
}