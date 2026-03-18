#include <iostream>

// TIP 코드를 <b>Run</b>하려면 <shortcut actionId="Run"/>을(를) 누르거나 여백에서 <icon src="AllIcons.Actions.Execute"/> 아이콘을 클릭하세요.
int main() {
    // TIP <b>lang</b> 변수 이름에 캐럿이 있을 때 <shortcut actionId="RenameElement"/>을(를) 누르면 CLion에서 이름을 변경하는 방법이 표시됩니다.
    auto lang = "C++";
    std::cout << "Hello and welcome to " << lang << "!\n";

    for (int i = 1; i <= 5; i++) {
        // TIP 코드 디버그를 시작하려면 <shortcut actionId="Debug"/>을(를) 누르세요. <icon src="AllIcons.Debugger.Db_set_breakpoint"/> 중단점 하나가 자동으로 설정되었습니다. <shortcut actionId="ToggleLineBreakpoint"/>을(를) 누르면 언제든지 중단점을 추가할 수 있습니다.
        std::cout << "i = " << i << std::endl;
    }

    return 0;
    // TIP <a href="https://www.jetbrains.com/help/clion/">jetbrains.com/help/clion/</a>에서 CLion 도움말을 참조하세요. 또한, 메인 메뉴의 '도움말 | IDE 기능 알아보기'를 선택하여 CLion에 관한 대화형 강의를 이용할 수도 있습니다.
}