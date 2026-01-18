//
//  AccessibilityService.swift
//  polar-bear
//
//  Created by Codex.
//

import AppKit
import ApplicationServices

final class AccessibilityService {
    func ensureProcessTrusted(prompt: Bool) -> Bool {
        let key = kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String
        let options = [key: prompt] as CFDictionary
        return AXIsProcessTrustedWithOptions(options)
    }

    func openAccessibilitySettings() {
        guard let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility") else {
            return
        }
        NSWorkspace.shared.open(url)
    }

    func focusedElement() -> AXUIElement? {
        let systemWide = AXUIElementCreateSystemWide()
        var focused: CFTypeRef?
        let error = AXUIElementCopyAttributeValue(systemWide, kAXFocusedUIElementAttribute as CFString, &focused)
        guard error == .success, let element = focused else {
            return nil
        }
        guard CFGetTypeID(element) == AXUIElementGetTypeID() else {
            return nil
        }
        return unsafeBitCast(element, to: AXUIElement.self)
    }

    func readText(from element: AXUIElement) -> String? {
        if isSecureTextField(element) {
            NSSound.beep()
            return nil
        }

        if let value = copyStringValue(from: element, attribute: kAXValueAttribute as CFString) {
            return value
        }

        if let selected = copyStringValue(from: element, attribute: kAXSelectedTextAttribute as CFString),
           !selected.isEmpty {
            return selected
        }

        if selectAllText(in: element),
           let selected = copyStringValue(from: element, attribute: kAXSelectedTextAttribute as CFString),
           !selected.isEmpty {
            return selected
        }

        if let clipboardText = copyTextViaKeyboard() {
            return clipboardText
        }

        NSSound.beep()
        return nil
    }

    func writeText(_ text: String, to element: AXUIElement) -> Bool {
        if AXUIElementSetAttributeValue(element, kAXValueAttribute as CFString, text as CFTypeRef) == .success {
            return true
        }

        if selectAllText(in: element),
           AXUIElementSetAttributeValue(element, kAXSelectedTextAttribute as CFString, text as CFTypeRef) == .success {
            return true
        }

        return AXUIElementSetAttributeValue(element, kAXSelectedTextAttribute as CFString, text as CFTypeRef) == .success
    }

    func pasteTextViaKeyboard(_ text: String) {
        let pasteboard = NSPasteboard.general
        let previousItems = pasteboard.pasteboardItems
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)

        sendKeyCombo(keyCode: 0, flags: .maskCommand) // Command + A
        sendKeyCombo(keyCode: 9, flags: .maskCommand) // Command + V

        restorePasteboardItems(previousItems)
    }

    func moveCaretToEnd(in element: AXUIElement, fallbackLength: Int) -> Bool {
        let length = numberOfCharacters(in: element) ?? fallbackLength
        guard length >= 0 else { return false }
        var range = CFRange(location: length, length: 0)
        guard let rangeValue = AXValueCreate(.cfRange, &range) else { return false }
        return AXUIElementSetAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, rangeValue) == .success
    }

    func moveCaretToEndViaKeyboard() {
        sendKeyCombo(keyCode: 125, flags: .maskCommand) // Command + Down
        sendKeyCombo(keyCode: 124, flags: .maskCommand) // Command + Right
    }

    private func isSecureTextField(_ element: AXUIElement) -> Bool {
        if let subrole = copyAttributeString(from: element, attribute: kAXSubroleAttribute as CFString),
           subrole == (kAXSecureTextFieldSubrole as String) {
            return true
        }

        return false
    }

    private func copyAttributeString(from element: AXUIElement, attribute: CFString) -> String? {
        var value: CFTypeRef?
        let error = AXUIElementCopyAttributeValue(element, attribute, &value)
        guard error == .success else { return nil }
        return value as? String
    }

    private func copyStringValue(from element: AXUIElement, attribute: CFString) -> String? {
        var value: CFTypeRef?
        let error = AXUIElementCopyAttributeValue(element, attribute, &value)
        guard error == .success, let value else { return nil }
        if let stringValue = value as? String {
            return stringValue
        }
        if let attributedValue = value as? NSAttributedString {
            return attributedValue.string
        }
        return nil
    }

    private func selectAllText(in element: AXUIElement) -> Bool {
        guard let count = numberOfCharacters(in: element) else { return false }
        guard count > 0 else { return false }
        var range = CFRange(location: 0, length: count)
        guard let rangeValue = AXValueCreate(.cfRange, &range) else { return false }
        return AXUIElementSetAttributeValue(element, kAXSelectedTextRangeAttribute as CFString, rangeValue) == .success
    }

    private func numberOfCharacters(in element: AXUIElement) -> Int? {
        var countValue: CFTypeRef?
        let countError = AXUIElementCopyAttributeValue(element, kAXNumberOfCharactersAttribute as CFString, &countValue)
        guard countError == .success else { return nil }

        if let number = countValue as? NSNumber {
            return number.intValue
        }
        if let intValue = countValue as? Int {
            return intValue
        }
        return nil
    }

    private func copyTextViaKeyboard() -> String? {
        let pasteboard = NSPasteboard.general
        let previousItems = pasteboard.pasteboardItems
        let previousChangeCount = pasteboard.changeCount

        sendKeyCombo(keyCode: 0, flags: .maskCommand) // Command + A
        sendKeyCombo(keyCode: 8, flags: .maskCommand) // Command + C

        let didChange = waitForPasteboardChange(from: previousChangeCount)
        let text = didChange ? pasteboard.string(forType: .string) : nil
        restorePasteboardItems(previousItems)
        return text
    }

    private func waitForPasteboardChange(from changeCount: Int) -> Bool {
        let deadline = Date().addingTimeInterval(0.3)
        while Date() < deadline {
            if NSPasteboard.general.changeCount != changeCount {
                return true
            }
            RunLoop.current.run(mode: .default, before: Date().addingTimeInterval(0.01))
        }
        return false
    }

    private func restorePasteboardItems(_ items: [NSPasteboardItem]?) {
        guard let items else { return }
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.2) {
            let pasteboard = NSPasteboard.general
            pasteboard.clearContents()
            pasteboard.writeObjects(items)
        }
    }

    private func sendKeyCombo(keyCode: CGKeyCode, flags: CGEventFlags) {
        guard let source = CGEventSource(stateID: .combinedSessionState) else { return }
        let keyDown = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: true)
        keyDown?.flags = flags
        let keyUp = CGEvent(keyboardEventSource: source, virtualKey: keyCode, keyDown: false)
        keyUp?.flags = flags
        keyDown?.post(tap: .cghidEventTap)
        keyUp?.post(tap: .cghidEventTap)
    }
}
