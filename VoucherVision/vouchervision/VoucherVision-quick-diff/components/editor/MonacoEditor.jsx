import React, { useState, useRef, useEffect } from "react";
import { DiffEditor } from "@monaco-editor/react";
import DiffModeMenu from "./DiffModeMenu";
import InputEditor from "./InputEditor";
import CTAButton from "../button/CTAButton";
import { PANEL, DIFF_MODE, LANGUAGE } from "../../utils/Constants";
import LanguageComboBox from "./LanguageComboBox";
import { defaultLeft, defaultRight } from "./Utils";

const diffModes = [
  { id: 1, mode: DIFF_MODE.SIDE_BY_SIDE },
  { id: 2, mode: DIFF_MODE.INLINE },
];

const defaultLanguage = {
  id: 1,
  name: LANGUAGE.JSON,
}

const MonacoEditor = () => {
  // const [leftInput, setLeftInput] = useState(defaultLeft);
  // const [rightInput, setRightInput] = useState(defaultRight);
  const [leftText, setLeftText] = useState('');
  const [rightText, setRightText] = useState('');
  
  const [numLines, setNumLines] = useState(25); // Default number of lines
  const lineHeight = 19; // Monaco Editor's approximate line height in pixels
  const editorHeight = `${numLines * lineHeight}px`; // Calculate height based on number of lines

  const [diffObj, setDiffObj] = useState({
    left: defaultLeft,
    right: defaultRight,
  });

  const [showDiffEditor, setShowDiffEditor] = useState(false);
  const [diffConfig, setDiffConfig] = useState({
    diffMode: diffModes[0],
    language: defaultLanguage,
  });

  const [selectedDiffMode, setSelectedDiffMode] = useState(diffModes[0]);

  const handleShowDiff = () => {
    setShowDiffEditor((prev) => !prev);
    setDiffObj({
      left: leftText,
      right: rightText,
    });
  };

  const handleClipboard = async (panel) => {
    const textInput = await navigator.clipboard.readText();

    switch (panel) {
      case PANEL.LEFT:
        setLeftInput(textInput);
        break;
      case PANEL.RIGHT:
        setRightInput(textInput);
        break;
      default:
        break;
    }
  };


  const handleClear = (panel) => {
    switch (panel) {
      case PANEL.LEFT:
        setLeftInput("");
        break;
      case PANEL.RIGHT:
        setRightInput("");
        break;
      default:
        break;
    }
  };

  const handleFavoriteSelection = (side) => {
    if (side === PANEL.LEFT) {
      setFavorite(leftInput);
    } else if (side === PANEL.RIGHT) {
      setFavorite(rightInput);
    }
  };

  const handleInputChange = (panel, value) => {
    switch (panel) {
      case PANEL.LEFT:
        setLeftInput(value);
        break;
      case PANEL.RIGHT:
        setRightInput(value);
        break;
      default:
        break;
    }
  };

  const languageComboBoxDidSelect = (newLanguage) => {
    setDiffConfig({
      ...diffConfig,
      language: newLanguage
    })
  }

  useEffect(() => {
    // Parse the query parameters
    const params = new URLSearchParams(window.location.search);
    
    const left = params.get('left');
    const right = params.get('right');
    
    try {
      // Decode and parse the JSON strings
      if (left) {
        setLeftText(JSON.stringify(JSON.parse(decodeURIComponent(left)), null, 2)); // Pretty print JSON
      }
      if (right) {
        setRightText(JSON.stringify(JSON.parse(decodeURIComponent(right)), null, 2)); // Pretty print JSON
      }
    } catch (error) {
      console.error("Error parsing JSON from URL:", error);
    }
  }, []);

  return (
    <div className="w-full flex flex-col gap-4">
      <div className="w-full flex flex-row justify-between items-end">
        <div className="flex flex-row gap-4 items-center">
          {/* Your existing components */}
          <LanguageComboBox
            selectedLanguage={diffConfig.language}
            setSelectedLanguage={languageComboBoxDidSelect}
          />
          <DiffModeMenu
            selected={selectedDiffMode}
            setSelected={setSelectedDiffMode}
            diffModes={diffModes}
          />
        </div>
        <div className="flex flex-row items-center gap-2">
          <label htmlFor="lineSlider">Visible Lines:</label>
          <input
            id="lineSlider"
            type="range"
            min="10" // Minimum 10 lines
            max="50" // Maximum 50 lines
            className="w-32 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            value={numLines} // Use numLines here instead of editorHeight
            onChange={(e) => setNumLines(Number(e.target.value))} // Convert to number
          />
          <span>{numLines} lines</span> {/* Display the number of lines */}
        </div>
        <div className="flex flex-row">
          <CTAButton handleOnClick={handleShowDiff}>
            {showDiffEditor ? "Edit Input" : "Show Diff"}
          </CTAButton>
        </div>
      </div>

      {showDiffEditor ? (
        <div className="w-full editorContainer">
          <DiffEditor
            className="w-full border border-gray-200 shadow-md"
            height={editorHeight} // Apply the calculated height
            language={diffConfig.language.name}
            theme="vs-light"
            original={diffObj.left}
            modified={diffObj.right}
            options={{
              renderSideBySide: selectedDiffMode.mode === DIFF_MODE.SIDE_BY_SIDE,
              scrollbar: {
                horizontalScrollbarSize: 8,
                verticalScrollbarSize: 8,
              },
              readOnly: true,
              diffWordWrap: "on",
              scrollBeyondLastLine: false,
            }}
          />
        </div>
      ) : (
        <div className="flex flex-col md:flex-row w-full editorContainer">
          <InputEditor
            // value={leftInput}
            value={leftText}
            onChange={setLeftText}
            language={diffConfig.language.name}
            height={editorHeight}
            styling={{ justifyContent: "flex-start" }}
            // onFavorite={() => handleFavoriteSelection(PANEL.LEFT)} // Add favorite callback
          />
          <div className="w-4 my-2 md:my-0"></div>
          <InputEditor
            // value={rightInput}
            value={rightText}
            onChange={setRightText}
            language={diffConfig.language.name}
            height={editorHeight}
            styling={{ justifyContent: "flex-end" }}
            // onFavorite={() => handleFavoriteSelection(PANEL.RIGHT)} // Add favorite callback

          />
        </div>
      )}
    </div>
  );
};

export default MonacoEditor;
