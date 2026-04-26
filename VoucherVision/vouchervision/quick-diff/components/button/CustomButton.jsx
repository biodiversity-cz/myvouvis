import React from 'react'

const CustomButton = ({ children, handleOnClick }) => {
    return (
      <button
        className="px-2.5 h-6 text-xs rounded text-white bg-[#1a1523] customBtn"
        onClick={handleOnClick}
      >
        {children}
      </button>
    );
  };

export default CustomButton