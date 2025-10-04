'use client'

import { useState } from 'react'

interface SearchFilters {
  band: string
  callsignHash: string
  startDate: string
  endDate: string
  propagationMode: string
  minSNR: number
}

interface Props {
  filters: SearchFilters
  onFiltersChange: (filters: SearchFilters) => void
  onSearch: () => void
  loading: boolean
}

export default function QASampleSearch({ filters, onFiltersChange, onSearch, loading }: Props) {
  const [isAdvanced, setIsAdvanced] = useState(false)

  const handleFilterChange = (field: keyof SearchFilters, value: any) => {
    onFiltersChange({
      ...filters,
      [field]: value
    })
  }

  const handleReset = () => {
    onFiltersChange({
      band: '',
      callsignHash: '',
      startDate: '',
      endDate: '',
      propagationMode: '',
      minSNR: 0
    })
  }

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-semibold text-white">Search QA Samples</h2>
        <button
          onClick={() => setIsAdvanced(!isAdvanced)}
          className="text-cascade-accent hover:text-cascade-blue transition-colors"
        >
          {isAdvanced ? 'Simple' : 'Advanced'} Search
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {/* Band Selection */}
        <div>
          <label className="block text-gray-400 text-sm mb-1">Band</label>
          <select
            value={filters.band}
            onChange={(e) => handleFilterChange('band', e.target.value)}
            className="w-full bg-gray-700 text-white rounded px-3 py-2"
          >
            <option value="">All Bands</option>
            <option value="160m">160m</option>
            <option value="80m">80m</option>
            <option value="40m">40m</option>
            <option value="30m">30m</option>
            <option value="20m">20m</option>
            <option value="17m">17m</option>
            <option value="15m">15m</option>
            <option value="12m">12m</option>
            <option value="10m">10m</option>
          </select>
        </div>

        {/* Callsign Hash */}
        <div>
          <label className="block text-gray-400 text-sm mb-1">Callsign Hash</label>
          <input
            type="text"
            value={filters.callsignHash}
            onChange={(e) => handleFilterChange('callsignHash', e.target.value)}
            placeholder="Hash prefix..."
            className="w-full bg-gray-700 text-white rounded px-3 py-2"
          />
        </div>

        {/* Start Date */}
        <div>
          <label className="block text-gray-400 text-sm mb-1">Start Date</label>
          <input
            type="datetime-local"
            value={filters.startDate}
            onChange={(e) => handleFilterChange('startDate', e.target.value)}
            className="w-full bg-gray-700 text-white rounded px-3 py-2"
          />
        </div>

        {/* End Date */}
        <div>
          <label className="block text-gray-400 text-sm mb-1">End Date</label>
          <input
            type="datetime-local"
            value={filters.endDate}
            onChange={(e) => handleFilterChange('endDate', e.target.value)}
            className="w-full bg-gray-700 text-white rounded px-3 py-2"
          />
        </div>

        {/* Advanced Filters */}
        {isAdvanced && (
          <>
            {/* Propagation Mode */}
            <div>
              <label className="block text-gray-400 text-sm mb-1">Propagation Mode</label>
              <select
                value={filters.propagationMode}
                onChange={(e) => handleFilterChange('propagationMode', e.target.value)}
                className="w-full bg-gray-700 text-white rounded px-3 py-2"
              >
                <option value="">Any Mode</option>
                <option value="F2">F2 Layer</option>
                <option value="Es">Sporadic E</option>
                <option value="TEP">Trans-Equatorial</option>
                <option value="EME">Earth-Moon-Earth</option>
                <option value="MS">Meteor Scatter</option>
                <option value="NVIS">NVIS</option>
                <option value="GW">Ground Wave</option>
              </select>
            </div>

            {/* Minimum SNR */}
            <div>
              <label className="block text-gray-400 text-sm mb-1">
                Min SNR: {filters.minSNR} dB
              </label>
              <input
                type="range"
                min="-30"
                max="30"
                value={filters.minSNR}
                onChange={(e) => handleFilterChange('minSNR', Number(e.target.value))}
                className="w-full"
              />
            </div>

            {/* Space Weather Condition */}
            <div>
              <label className="block text-gray-400 text-sm mb-1">Space Weather</label>
              <select
                className="w-full bg-gray-700 text-white rounded px-3 py-2"
              >
                <option value="">Any Condition</option>
                <option value="quiet">Quiet</option>
                <option value="unsettled">Unsettled</option>
                <option value="active">Active</option>
                <option value="minor_storm">Minor Storm</option>
                <option value="major_storm">Major Storm</option>
                <option value="severe_storm">Severe Storm</option>
              </select>
            </div>

            {/* Signal Type */}
            <div>
              <label className="block text-gray-400 text-sm mb-1">Signal Type</label>
              <select
                className="w-full bg-gray-700 text-white rounded px-3 py-2"
              >
                <option value="">All Types</option>
                <option value="ft8">FT8</option>
                <option value="wspr">WSPR</option>
                <option value="cw">CW</option>
                <option value="ssb">SSB</option>
                <option value="qrn">QRN Only</option>
              </select>
            </div>
          </>
        )}
      </div>

      {/* Action Buttons */}
      <div className="flex justify-between items-center mt-6">
        <div className="text-sm text-gray-400">
          {isAdvanced && (
            <span>Advanced filters active</span>
          )}
        </div>
        <div className="flex gap-3">
          <button
            onClick={handleReset}
            className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors"
          >
            Reset
          </button>
          <button
            onClick={onSearch}
            disabled={loading}
            className={`px-6 py-2 rounded transition-colors ${
              loading
                ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                : 'bg-cascade-accent hover:bg-cascade-blue text-white'
            }`}
          >
            {loading ? 'Searching...' : 'Search'}
          </button>
        </div>
      </div>

      {/* Quick Filters */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <p className="text-sm text-gray-400 mb-2">Quick Filters:</p>
        <div className="flex flex-wrap gap-2">
          <QuickFilterButton
            label="Last 24h"
            onClick={() => {
              const now = new Date()
              const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000)
              handleFilterChange('startDate', yesterday.toISOString().slice(0, 16))
              handleFilterChange('endDate', now.toISOString().slice(0, 16))
            }}
          />
          <QuickFilterButton
            label="High SNR (>10dB)"
            onClick={() => handleFilterChange('minSNR', 10)}
          />
          <QuickFilterButton
            label="20m Band"
            onClick={() => handleFilterChange('band', '20m')}
          />
          <QuickFilterButton
            label="FT8 Only"
            onClick={() => {
              // This would need to be added to filters
              console.log('Filter by FT8')
            }}
          />
          <QuickFilterButton
            label="Storm Events"
            onClick={() => {
              // This would need to be added to filters
              console.log('Filter by storm events')
            }}
          />
        </div>
      </div>
    </div>
  )
}

function QuickFilterButton({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded transition-colors"
    >
      {label}
    </button>
  )
}